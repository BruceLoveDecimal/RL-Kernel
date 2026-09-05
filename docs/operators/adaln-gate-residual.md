# AdaLN gated residual

Computes `y = x + gate * sublayer_out` for the image/text attention and MLP residuals in Qwen-Image. The sublayer and AdaLN modulation are computed by the caller. This operation allocates a new output and does not modify inputs.

```python
from rl_engine.kernels.adaln_gate_residual import adaln_gate_residual

trace = {}
y = adaln_gate_residual(x, gate, sublayer_out, backend="auto", trace=trace)
y.backward(dy)
```

`x` and `sublayer_out` share `[B,S,D]`, device and FP32/BF16 dtype. Gate is `[B,D]`, `[B,1,D]` or `[B,S,D]`, in the same dtype or FP32. The first two shapes broadcast over tokens. Inputs may be strided views; the wrapper normalizes their layout. Output has x.dtype, and each input gradient has its input's dtype and shape. B and D must be positive; S may be zero. FP16 and second derivatives are unsupported.

## Arithmetic contract

`adaln_gate_residual.fp32_separate_mul_add.chunk256_tree.v1` uses separate rounded FP32 multiplication and addition, followed by one final output cast. Intermediate products and gate-gradient partials are never BF16. FMA, FTZ, fast-math, TF32 and atomic accumulation are prohibited.

The broadcast gate gradient groups tokens into fixed 256-token chunks. Each chunk is reduced by eight levels of adjacent pairs, padding missing leaves with positive zero. Chunk results are then added to a positive-zero accumulator in ascending chunk order. CUDA evaluates the same tree with a binary carry stack; Triton explicitly gathers adjacent pairs. This fixes arithmetic independently of launch geometry and feature tiling.

CUDA and Triton require NVIDIA SM80 or later. Explicit backend selection raises on missing capabilities. Auto selection tries CUDA, Triton and PyTorch on NVIDIA GPUs, with a warning and trace for each capability fallback. Runtime execution errors propagate. CPU uses PyTorch. Other device families are not certified by this implementation.

Trace includes requested/actual forward and backward backends, contract, strides, copies, launch configuration, arithmetic policy, source/build and Triton PTX fingerprints. A trace describes execution; a passing validation report is required for certification. A mutable trace dictionary must belong to one invocation and must not be reused before that invocation's backward completes.

## Invariance scope

A sample's outputs and local VJPs do not depend on its batch companions. Elementwise outputs are also independent of token position/count when their input values and gate are unchanged. Broadcast dgate depends on all active tokens: removing tokens is a different derivative. Arbitrary external autograd graphs, shared gate views and shared ancestors can introduce additional reductions outside this operator's local VJP contract.

Finite FP32/BF16 results, signed zeros and subnormals are compared as raw bits. NaN/Inf propagation is tested separately; NaN payload identity is not promised across backends. CUDA is compared against an independent CPU FP32 oracle, and BF16 outputs against that oracle cast once to BF16.

## Build and validation

Build without `KERNEL_ALIGN_USE_FAST_MATH`; the CUDA extension rejects this setting for this operator. The build adds `--ftz=false`; explicit round-to-nearest multiplication/addition intrinsics prevent FMA contraction in this operator without changing other kernels' contraction flags. The CUDA source, effective flags and PyTorch/CUDA versions contribute to its build fingerprint.

```bash
KERNEL_ALIGN_USE_FAST_MATH=0 python -m pip install -e . --no-build-isolation
python -m pytest tests/test_adaln_gate_residual.py tests/test_adaln_gate_residual_invariance.py tests/test_adaln_gate_residual_dispatch.py -q -rs
python scripts/validate_adaln_gate_residual.py --backend all --suite strict --require-gpu --output artifacts/adaln-strict.json
python benchmarks/benchmark_adaln_gate_residual.py --backend all --suite qwen-image --output artifacts/adaln-benchmark.json
```

The strict runner covers image sequences 4096, 6889 and 6032 at D=3072, B=1/2/4, three gate shapes, FP32/BF16 and FP32 gate with BF16 activations, plus text cases and focused edge/invariance tests. GPU absence fails `--require-gpu`. CPU-only smoke is available with `--backend pytorch --suite smoke`.

Benchmarks exclude compilation, use 20 warmups and 100 measured iterations, and report median/p95 latency and extra peak allocated memory for forward, backward and forward+backward. They measure the public API including layout normalization. The eager low-precision expression is separately labeled because its intermediate rounding and gate-gradient reduction may differ.

Validation status and the actual machine/software inventory belong in the generated reports. No GPU correctness or speedup is claimed merely by installing this implementation.

## Initial validation record (2026-09-05)

The implementation is awaiting GPU compilation/acceptance. Local Intel macOS testing used an isolated Python 3.12.14 / PyTorch 2.2.2 CPU environment (the available Intel macOS wheel); this is a preliminary oracle check, not a claim that the whole repository supports PyTorch below its declared minimum.

- Focused operator tests plus existing registry/dispatch regression: **150 passed, 211 GPU skips**.
- CPU strict smoke: all nine dtype/gate combinations passed raw-bit forward and local VJP comparison.
- General `check_operator.py` BF16 forward/backward: passed.
- `--require-gpu` on a CPU host: nonzero exit and `passed: false`, as required.
- New Python source lint/format checks and strict MkDocs build: passed.

The prepared GPU environment has RTX 5090 32GB (SM120), driver 595.58.03, Python 3.12.3, PyTorch 2.13.0+cu130 and Triton 3.7.1. CUDA 13.0.88 compiler/runtime and CCCL 13.0.85 were installed separately, matching PyTorch's CUDA major version without replacing the existing CUDA 12.8 toolkit or Python environment.

GPU validation remains pending. No GPU correctness, sanitizer, timing, cold-JIT or cross-SM result is claimed yet. Cross-SM validation additionally needs a different NVIDIA architecture, such as SM80 or SM90; one RTX 5090 cannot establish that criterion.
