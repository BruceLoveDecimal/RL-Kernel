# SPDX-License-Identifier: Apache-2.0
"""Strict FP32 gated residual with observable, device-aware backend selection."""

from __future__ import annotations

import hashlib
import importlib
import json
from functools import lru_cache
from pathlib import Path

import torch
from torch.autograd.function import once_differentiable

from rl_engine.utils.logger import logger

CONTRACT = "adaln_gate_residual.fp32_separate_mul_add.chunk256_tree.v1"
CHUNK = 256
_BACKENDS = {
    "pytorch": "pytorch.modulation.adaln_gate_residual.NativeAdaLNGateResidualOp",
    "cuda": "cuda.modulation.adaln_gate_residual.CudaAdaLNGateResidualOp",
    "triton": "triton.modulation.adaln_gate_residual.TritonAdaLNGateResidualOp",
}


class BackendUnavailable(RuntimeError):
    """A capability gap, never a kernel execution failure."""


def validate(x, gate, sublayer_out):
    if not all(isinstance(t, torch.Tensor) for t in (x, gate, sublayer_out)):
        raise TypeError("x, gate and sublayer_out must be tensors")
    if x.ndim != 3 or x.shape != sublayer_out.shape:
        raise ValueError("x and sublayer_out must have the same [B,S,D] shape")
    b, s, d = x.shape
    if b < 1 or d < 1:
        raise ValueError("B and D must be positive")
    if tuple(gate.shape) not in ((b, d), (b, 1, d), (b, s, d)):
        raise ValueError("gate must have shape [B,D], [B,1,D] or [B,S,D]")
    if x.dtype not in (torch.float32, torch.bfloat16) or sublayer_out.dtype != x.dtype:
        raise TypeError("x and sublayer_out must have matching FP32 or BF16 dtype")
    if gate.dtype not in (x.dtype, torch.float32):
        raise TypeError("gate must have x.dtype or FP32 dtype")
    if x.device != gate.device or x.device != sublayer_out.device:
        raise ValueError("all inputs must be on the same device")
    if any(t.layout != torch.strided for t in (x, gate, sublayer_out)):
        raise ValueError("only strided tensors are supported")
    return gate.ndim == 2 or gate.shape[1] == 1


@lru_cache(maxsize=None)
def source_fingerprint(module_file):
    digest = hashlib.sha256()
    for path in (Path(__file__), Path(module_file)):
        digest.update(path.read_bytes())
    return digest.hexdigest()


class _ResidualFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, gate, sublayer_out, op, trace):
        ctx.op, ctx.trace = op, trace
        ctx.gate_shape = gate.shape
        ctx.shape = x.shape
        ctx.dtype = x.dtype
        ctx.gate_dtype = gate.dtype
        ctx.broadcast = validate(x, gate, sublayer_out)
        # Save only inputs needed by requested VJPs. Saved originals retain version checks.
        ctx.save_for_backward(
            gate if ctx.needs_input_grad[2] else None,
            sublayer_out if ctx.needs_input_grad[1] else None,
        )
        return op.forward_raw(x.contiguous(), gate.contiguous(), sublayer_out.contiguous())

    @staticmethod
    @once_differentiable
    def backward(ctx, dy):
        gate, s = ctx.saved_tensors
        need_x, need_g, need_s = ctx.needs_input_grad[:3]
        dg = ds = None
        if need_g or need_s:
            dg, ds = ctx.op.backward_raw(
                dy.contiguous(),
                gate,
                s,
                ctx.gate_shape,
                ctx.gate_dtype,
                ctx.broadcast,
                need_g,
                need_s,
            )
        if ctx.trace is not None:
            ctx.trace["backward_backend"] = ctx.op.backend
            ctx.trace["backward_executed"] = True
            ctx.trace["backward_layout_copy"] = not dy.is_contiguous()
            ctx.trace.update(ctx.op.build_info())
        return dy if need_x else None, dg, ds, None, None


class ResidualOp:
    """Shared validation/autograd; subclasses own numerical kernels."""

    op_class = "elementwise"
    backend = "pytorch"

    def check_available(self, device):
        pass

    def build_info(self):
        module = importlib.import_module(type(self).__module__)
        parameters = {"torch": torch.__version__, "contract": CONTRACT}
        source = source_fingerprint(module.__file__)
        fingerprint = hashlib.sha256((source + json.dumps(parameters)).encode()).hexdigest()
        return {
            "source_fingerprint": source,
            "kernel_fingerprint": fingerprint,
            "build_parameters": parameters,
        }

    def forward(self, x, gate, sublayer_out, *, trace=None):
        broadcast = validate(x, gate, sublayer_out)
        self.check_available(x.device)
        if trace is not None:
            trace.setdefault("requested_backend", self.backend)
            trace.setdefault("fallback_reasons", [])
            trace.update(
                {
                    "op": "adaln_gate_residual",
                    "contract": CONTRACT,
                    "forward_backend": self.backend,
                    "backward_backend": None,
                    "backward_executed": False,
                    "device": str(x.device),
                    "shape": list(x.shape),
                    "gate_shape": list(gate.shape),
                    "strides": [list(t.stride()) for t in (x, gate, sublayer_out)],
                    "dtypes": [str(t.dtype) for t in (x, gate, sublayer_out)],
                    "gate_mode": "broadcast" if broadcast else "elementwise",
                    "layout_copy": [not t.is_contiguous() for t in (x, gate, sublayer_out)],
                    "accumulator": "float32",
                    "chunk": CHUNK,
                    "reduction_order": "adjacent_pairs_8_levels_then_ascending_chunk_left_fold",
                    "fma": False,
                    "ftz": False,
                    "fast_math": False,
                    "tf32": False,
                    "split_k": False,
                    "stream_k": False,
                    "atomic_accumulation": False,
                    "tolerance_profile": None,
                    "certification": "requires_validation_report",
                    "sm": (
                        list(torch.cuda.get_device_capability(x.device))
                        if x.device.type == "cuda" and torch.version.hip is None
                        else None
                    ),
                }
            )
            trace.update(self.build_info())
        result = _ResidualFunction.apply(x, gate, sublayer_out, self, trace)
        if trace is not None:
            trace.update(self.build_info())
        return result

    __call__ = forward

    def forward_fp32(self, x, gate, sublayer_out):
        return self.forward(x.float(), gate.float(), sublayer_out.float())


def load_backend(name):
    if name not in _BACKENDS:
        raise ValueError(f"unknown adaln backend {name!r}; expected auto/cuda/triton/pytorch")
    module, cls = _BACKENDS[name].rsplit(".", 1)
    try:
        return getattr(importlib.import_module(f"rl_engine.kernels.ops.{module}"), cls)()
    except ModuleNotFoundError as exc:
        if exc.name != "triton" and not (exc.name or "").startswith("triton."):
            raise
        raise BackendUnavailable("Triton is not installed") from exc


def adaln_gate_residual(x, gate, sublayer_out, *, backend="auto", trace=None):
    """Compute x + gate * sublayer_out; explicit backends never fall back."""
    validate(x, gate, sublayer_out)
    if trace is not None:
        trace.clear()
        trace.update(requested_backend=backend, fallback_reasons=[])
    names = [backend]
    if backend == "auto":
        names = (
            ["cuda", "triton", "pytorch"]
            if (x.device.type == "cuda" and torch.version.hip is None)
            else ["pytorch"]
        )
    for name in names:
        try:
            op = load_backend(name)
            op.check_available(x.device)
        except BackendUnavailable as exc:
            if backend != "auto":
                raise
            reason = f"{name}: {exc}"
            logger.warning("adaln_gate_residual fallback: %s", reason)
            if trace is not None:
                trace["fallback_reasons"].append(reason)
            continue
        # Execution errors intentionally escape; never retry a failed CUDA launch.
        return op(x, gate, sublayer_out, trace=trace)
    raise BackendUnavailable("no adaln_gate_residual backend is available")
