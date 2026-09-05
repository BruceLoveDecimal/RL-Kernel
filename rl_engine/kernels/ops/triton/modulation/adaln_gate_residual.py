# SPDX-License-Identifier: Apache-2.0
"""NVIDIA Triton kernels with an explicit adjacent-pair reduction DAG."""

import hashlib

import torch
import triton
import triton.language as tl

from rl_engine.kernels.adaln_gate_residual import BackendUnavailable, ResidualOp


@triton.jit
def _mul(a, b):
    return tl.inline_asm_elementwise(
        "mul.rn.f32 $0, $1, $2;",
        constraints="=f,f,f",
        args=[a, b],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )


@triton.jit
def _add(a, b):
    return tl.inline_asm_elementwise(
        "add.rn.f32 $0, $1, $2;",
        constraints="=f,f,f",
        args=[a, b],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )


@triton.jit
def _point(
    X,
    G,
    S,
    Y,
    DG,
    DS,
    N: tl.constexpr,
    SEQ: tl.constexpr,
    D: tl.constexpr,
    BROADCAST: tl.constexpr,
    BACKWARD: tl.constexpr,
    NEED_G: tl.constexpr,
    NEED_S: tl.constexpr,
    BLOCK: tl.constexpr,
):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = i < N
    gi = i // (SEQ * D) * D + i % D if BROADCAST else i
    v = tl.load(X + i, mask, other=0).to(tl.float32)
    if not BACKWARD:
        g = tl.load(G + gi, mask, other=0).to(tl.float32)
        s = tl.load(S + i, mask, other=0).to(tl.float32)
        tl.store(Y + i, _add(v, _mul(g, s)), mask)
    else:
        if NEED_S:
            g = tl.load(G + gi, mask, other=0).to(tl.float32)
            tl.store(DS + i, _mul(v, g), mask)
        if NEED_G and not BROADCAST:
            s = tl.load(S + i, mask, other=0).to(tl.float32)
            tl.store(DG + i, _mul(v, s), mask)


@triton.jit
def _partial(
    DY, S, P, SEQ: tl.constexpr, D: tl.constexpr, CHUNKS: tl.constexpr, FEATURE: tl.constexpr
):
    batch = tl.program_id(0)
    chunk = tl.program_id(1)
    d = tl.program_id(2) * FEATURE + tl.arange(0, FEATURE)
    t = chunk * 256 + tl.arange(0, 256)
    offset = (batch * SEQ + t[:, None]) * D + d[None, :]
    mask = (t[:, None] < SEQ) & (d[None, :] < D)
    dy = tl.load(DY + offset, mask, other=0).to(tl.float32)
    s = tl.load(S + offset, mask, other=0).to(tl.float32)
    # Fill absent leaves with +0 (not a masked product with an arbitrary sign).
    values = tl.where(mask, _mul(dy, s), 0.0)
    for level in tl.static_range(0, 8):
        even = tl.arange(0, 128 >> level)[:, None] * 2 + tl.zeros((1, FEATURE), tl.int32)
        left = tl.gather(values, even, axis=0)
        right = tl.gather(values, even + 1, axis=0)
        values = _add(left, right)
    root = tl.reshape(values, (FEATURE,))
    tl.store(P + (batch * CHUNKS + chunk) * D + d, root, d < D)


@triton.jit
def _finish(P, DG, D: tl.constexpr, CHUNKS: tl.constexpr, TOTAL: tl.constexpr, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    acc = tl.full((BLOCK,), 0, tl.float32)
    for j in range(CHUNKS):
        v = tl.load(P + (i // D * CHUNKS + j) * D + i % D, i < TOTAL, other=0)
        acc = _add(acc, v)
    tl.store(DG + i, acc, i < TOTAL)


class TritonAdaLNGateResidualOp(ResidualOp):
    backend = "triton"

    def __init__(self, *, num_warps=4, block=256, feature=16):
        if num_warps not in (4, 8) or block not in (128, 256, 512) or feature not in (8, 16, 32):
            raise ValueError("unsupported launch configuration")
        self.num_warps, self.block, self.feature = num_warps, block, feature
        self.compiled = {}

    def check_available(self, device):
        if torch.device(device).type != "cuda" or torch.version.hip is not None:
            raise BackendUnavailable("strict Triton v1 uses NVIDIA PTX")
        if torch.cuda.get_device_capability(device)[0] < 8:
            raise BackendUnavailable("v1 requires SM80+")

    def _record(self, name, compiled):
        self.compiled[name] = hashlib.sha256(compiled.asm["ptx"].encode()).hexdigest()

    def build_info(self):
        result = super().build_info()
        result.update(
            triton_version=triton.__version__,
            ptx_fingerprints=dict(self.compiled),
            launch={
                "num_warps": self.num_warps,
                "block": self.block,
                "feature": self.feature,
                "enable_fp_fusion": False,
            },
        )
        result["kernel_fingerprint"] = hashlib.sha256(
            (
                result["kernel_fingerprint"]
                + triton.__version__
                + repr(result["launch"])
                + repr(sorted(self.compiled.items()))
            ).encode()
        ).hexdigest()
        return result

    def forward_raw(self, x, gate, sublayer_out):
        out = torch.empty_like(x)
        if x.numel():
            compiled = _point[(triton.cdiv(x.numel(), self.block),)](
                x,
                gate,
                sublayer_out,
                out,
                out,
                out,
                x.numel(),
                x.shape[1],
                x.shape[2],
                gate.ndim == 2 or gate.shape[1] == 1,
                False,
                False,
                False,
                self.block,
                num_warps=self.num_warps,
                enable_fp_fusion=False,
            )
            self._record("forward", compiled)
        return out

    def backward_raw(self, dy, gate, s, gate_shape, gate_dtype, broadcast, need_g, need_s):
        dg = torch.empty(gate_shape, device=dy.device, dtype=gate_dtype) if need_g else None
        ds = torch.empty_like(dy) if need_s else None
        if not dy.numel():
            if dg is not None:
                dg.zero_()
            return dg, ds
        gate = gate.contiguous() if gate is not None else dy
        s = s.contiguous() if s is not None else dy
        if need_s or (need_g and not broadcast):
            compiled = _point[(triton.cdiv(dy.numel(), self.block),)](
                dy,
                gate,
                s,
                dy,
                dg if dg is not None else dy,
                ds if ds is not None else dy,
                dy.numel(),
                dy.shape[1],
                dy.shape[2],
                broadcast,
                True,
                need_g,
                need_s,
                self.block,
                num_warps=self.num_warps,
                enable_fp_fusion=False,
            )
            self._record("pointwise_backward", compiled)
        if need_g and broadcast:
            b, seq, d = dy.shape
            chunks = triton.cdiv(seq, 256)
            partial = torch.empty((b, chunks, d), dtype=torch.float32, device=dy.device)
            compiled = _partial[(b, chunks, triton.cdiv(d, self.feature))](
                dy,
                s,
                partial,
                seq,
                d,
                chunks,
                self.feature,
                num_warps=self.num_warps,
                enable_fp_fusion=False,
            )
            self._record("partial", compiled)
            compiled = _finish[(triton.cdiv(b * d, self.block),)](
                partial,
                dg,
                d,
                chunks,
                b * d,
                self.block,
                num_warps=self.num_warps,
                enable_fp_fusion=False,
            )
            self._record("finish", compiled)
        return dg, ds
