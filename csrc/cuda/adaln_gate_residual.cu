// SPDX-License-Identifier: Apache-2.0
// FP32 separate mul/add; adjacent 256-leaf tree then ascending chunk left fold.
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_bf16.h>

#ifdef __FAST_MATH__
#error "adaln_gate_residual requires fast-math disabled"
#endif
#ifndef ADALN_BUILD_FINGERPRINT
#define ADALN_BUILD_FINGERPRINT "unverified-build"
#endif

namespace {
constexpr int C = 256;

template <typename T, typename G>
__global__ void forward_kernel(const T* x, const G* g, const T* s, T* y,
                               int64_t n, int64_t seq, int64_t d, bool broadcast) {
    for (int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
         i < n; i += int64_t(gridDim.x) * blockDim.x) {
        int64_t gi = broadcast ? (i / (seq * d)) * d + i % d : i;
        float p = __fmul_rn(float(g[gi]), float(s[i]));
        y[i] = T(__fadd_rn(float(x[i]), p));
    }
}

template <typename T, typename G>
__global__ void pointwise_backward(const T* dy, const G* g, const T* s,
                                   G* dg, T* ds, int64_t n, int64_t seq, int64_t d,
                                   bool broadcast, bool need_g, bool need_s) {
    for (int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
         i < n; i += int64_t(gridDim.x) * blockDim.x) {
        float v = float(dy[i]);
        if (need_s) {
            int64_t gi = broadcast ? (i / (seq * d)) * d + i % d : i;
            ds[i] = T(__fmul_rn(v, float(g[gi])));
        }
        if (need_g && !broadcast) dg[i] = G(__fmul_rn(v, float(s[i])));
    }
}

template <typename T>
__global__ void partial_kernel(const T* dy, const T* s, float* partial,
                               int64_t total, int64_t seq, int64_t d, int64_t chunks) {
    for (int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
         i < total; i += int64_t(gridDim.x) * blockDim.x) {
        int64_t channel = i % d, chunk = (i / d) % chunks, batch = i / (d * chunks);
        // A binary carry stack evaluates exactly the adjacent-pair tree without
        // storing 256 leaves per lane. No scheduling-dependent shared reduction.
        float stack[9];
        float v = 0.0f;
        #pragma unroll 1
        for (int t = 0; t < C; ++t) {
            int64_t token = chunk * C + t;
            int64_t offset = (batch * seq + token) * d + channel;
            v = token < seq ? __fmul_rn(float(dy[offset]), float(s[offset])) : 0.0f;
            int count = t + 1;
            int level = 0;
            #pragma unroll 1
            while ((count & 1) == 0) {
                v = __fadd_rn(stack[level], v);
                count >>= 1;
                ++level;
            }
            stack[level] = v;
        }
        partial[i] = v;
    }
}

template <typename G>
__global__ void finish_kernel(const float* partial, G* dg, int64_t n,
                              int64_t d, int64_t chunks) {
    for (int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
         i < n; i += int64_t(gridDim.x) * blockDim.x) {
        float acc = 0.0f;
        for (int64_t j = 0; j < chunks; ++j)
            acc = __fadd_rn(acc, partial[(i / d * chunks + j) * d + i % d]);
        dg[i] = G(acc);
    }
}

void check_tensor(const torch::Tensor& t) {
    TORCH_CHECK(t.is_cuda() && t.is_contiguous(), "requires contiguous CUDA tensors");
    TORCH_CHECK(t.scalar_type() == at::kFloat || t.scalar_type() == at::kBFloat16,
                "requires FP32 or BF16");
}
bool check_shape(const torch::Tensor& x, at::IntArrayRef shape) {
    TORCH_CHECK(x.dim() == 3 && x.size(0) > 0 && x.size(2) > 0, "invalid [B,S,D]");
    bool broadcast = (shape.size() == 2 && shape[0] == x.size(0) && shape[1] == x.size(2)) ||
        (shape.size() == 3 && shape[0] == x.size(0) && shape[1] == 1 && shape[2] == x.size(2));
    TORCH_CHECK(broadcast || shape == x.sizes(), "invalid gate shape");
    return broadcast;
}
int grid(int64_t n, int threads, int cap) {
    return int(std::min<int64_t>((n + threads - 1) / threads, cap));
}
void check_launch(int threads, int cap) {
    TORCH_CHECK((threads == 128 || threads == 256) && cap > 0 && cap <= 65535,
                "threads must be 128/256 and grid cap in [1,65535]");
}
}  // namespace

torch::Tensor adaln_gate_residual_forward(torch::Tensor x, torch::Tensor gate,
                                         torch::Tensor s, int64_t threads, int64_t cap) {
    check_tensor(x); check_tensor(gate); check_tensor(s); check_launch(threads, cap);
    bool broadcast = check_shape(x, gate.sizes());
    TORCH_CHECK(x.sizes() == s.sizes() && x.scalar_type() == s.scalar_type(), "invalid s");
    TORCH_CHECK(x.device() == s.device() && x.device() == gate.device(), "device mismatch");
    TORCH_CHECK(gate.scalar_type() == x.scalar_type() || gate.scalar_type() == at::kFloat,
                "invalid gate dtype");
    c10::cuda::CUDAGuard guard(x.device());
    auto y = torch::empty_like(x);
    if (!x.numel()) return y;
    auto stream = at::cuda::getCurrentCUDAStream();
    int blocks = grid(x.numel(), threads, cap);
    #define FWD(T,G) forward_kernel<T,G><<<blocks,threads,0,stream>>>( \
        x.data_ptr<T>(),gate.data_ptr<G>(),s.data_ptr<T>(),y.data_ptr<T>(), \
        x.numel(),x.size(1),x.size(2),broadcast)
    if (x.scalar_type() == at::kFloat) { FWD(float,float); }
    else if (gate.scalar_type() == at::kFloat) { FWD(at::BFloat16,float); }
    else { FWD(at::BFloat16,at::BFloat16); }
    #undef FWD
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return y;
}

std::vector<torch::Tensor> adaln_gate_residual_backward(
    torch::Tensor dy, torch::Tensor gate, torch::Tensor s, std::vector<int64_t> gate_shape,
    bool need_g, bool need_s, int64_t threads, int64_t cap) {
    check_tensor(dy); check_tensor(gate); check_tensor(s); check_launch(threads, cap);
    bool broadcast = check_shape(dy, gate_shape);
    TORCH_CHECK(dy.device() == s.device() && dy.device() == gate.device(), "device mismatch");
    TORCH_CHECK(s.scalar_type() == dy.scalar_type(), "s dtype mismatch");
    TORCH_CHECK(gate.scalar_type() == dy.scalar_type() || gate.scalar_type() == at::kFloat,
                "gate dtype mismatch");
    if (need_g) TORCH_CHECK(s.sizes() == dy.sizes(), "s shape mismatch");
    if (need_s) TORCH_CHECK(gate.sizes() == at::IntArrayRef(gate_shape), "gate shape mismatch");
    c10::cuda::CUDAGuard guard(dy.device());
    auto dg = torch::empty(need_g ? gate_shape : std::vector<int64_t>{0}, gate.options());
    auto ds = torch::empty(need_s ? dy.sizes().vec() : std::vector<int64_t>{0}, dy.options());
    if (!dy.numel()) {
        if (need_g) dg.zero_();
        return {dg, ds};
    }
    int64_t d = dy.size(2), chunks = (dy.size(1) + C - 1) / C;
    auto stream = at::cuda::getCurrentCUDAStream();
    #define BWD(T,G) pointwise_backward<T,G><<<grid(dy.numel(),threads,cap),threads,0,stream>>>( \
        dy.data_ptr<T>(),gate.data_ptr<G>(),s.data_ptr<T>(),dg.data_ptr<G>(),ds.data_ptr<T>(), \
        dy.numel(),dy.size(1),d,broadcast,need_g,need_s)
    if (need_s || (need_g && !broadcast)) {
        if (dy.scalar_type() == at::kFloat) { BWD(float,float); }
        else if (gate.scalar_type() == at::kFloat) { BWD(at::BFloat16,float); }
        else { BWD(at::BFloat16,at::BFloat16); }
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
    #undef BWD
    if (need_g && broadcast) {
        auto partial = torch::empty({dy.size(0),chunks,d}, dy.options().dtype(at::kFloat));
        #define PART(T) partial_kernel<T><<<grid(partial.numel(),threads,cap),threads,0,stream>>>( \
            dy.data_ptr<T>(),s.data_ptr<T>(),partial.data_ptr<float>(), \
            partial.numel(),dy.size(1),d,chunks)
        if (dy.scalar_type() == at::kFloat) { PART(float); } else { PART(at::BFloat16); }
        #undef PART
        C10_CUDA_KERNEL_LAUNCH_CHECK();
        #define FIN(G) finish_kernel<G><<<grid(dg.numel(),threads,cap),threads,0,stream>>>( \
            partial.data_ptr<float>(),dg.data_ptr<G>(),dg.numel(),d,chunks)
        if (gate.scalar_type() == at::kFloat) { FIN(float); } else { FIN(at::BFloat16); }
        #undef FIN
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
    return {dg, ds};
}

std::string adaln_gate_residual_build_fingerprint() { return ADALN_BUILD_FINGERPRINT; }
