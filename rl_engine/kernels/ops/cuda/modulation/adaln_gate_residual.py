# SPDX-License-Identifier: Apache-2.0
import hashlib
from functools import lru_cache
from pathlib import Path

import torch

from rl_engine.kernels.adaln_gate_residual import BackendUnavailable, ResidualOp
from rl_engine.kernels.ops import base


@lru_cache(maxsize=None)
def _binary_fingerprint(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class CudaAdaLNGateResidualOp(ResidualOp):
    backend = "cuda"

    def __init__(self, *, threads=256, grid_cap=65535):
        self.threads, self.grid_cap = threads, grid_cap
        self._extension()

    @staticmethod
    def _extension():
        names = (
            "adaln_gate_residual_forward",
            "adaln_gate_residual_backward",
            "adaln_gate_residual_build_fingerprint",
        )
        if base._C is None or not all(hasattr(base._C, n) for n in names):
            raise BackendUnavailable("compiled adaln forward/backward/build symbols missing")
        if base._C.adaln_gate_residual_build_fingerprint() == "unverified-build":
            raise BackendUnavailable("extension missing strict build provenance")
        return base._C

    def check_available(self, device):
        self._extension()
        if torch.device(device).type != "cuda" or torch.version.hip is not None:
            raise BackendUnavailable("NVIDIA CUDA tensors required")
        if torch.cuda.get_device_capability(device)[0] < 8:
            raise BackendUnavailable("v1 requires SM80+ for native BF16")

    def build_info(self):
        result = super().build_info()
        result.update(
            cuda_build_fingerprint=self._extension().adaln_gate_residual_build_fingerprint(),
            launch={"threads": self.threads, "grid_cap": self.grid_cap},
        )
        binary_path = getattr(self._extension(), "__file__", None)
        result["binary_fingerprint"] = _binary_fingerprint(binary_path) if binary_path else None
        result["kernel_fingerprint"] = hashlib.sha256(
            (
                result["kernel_fingerprint"]
                + result["cuda_build_fingerprint"]
                + repr(result["launch"])
                + str(result["binary_fingerprint"])
            ).encode()
        ).hexdigest()
        return result

    def forward_raw(self, x, gate, sublayer_out):
        return self._extension().adaln_gate_residual_forward(
            x, gate, sublayer_out, self.threads, self.grid_cap
        )

    def backward_raw(self, dy, gate, s, gate_shape, gate_dtype, broadcast, need_g, need_s):
        g = (
            gate.contiguous()
            if gate is not None
            else torch.empty(0, device=dy.device, dtype=gate_dtype)
        )
        sub = s.contiguous() if s is not None else torch.empty(0, device=dy.device, dtype=dy.dtype)
        dg, ds = self._extension().adaln_gate_residual_backward(
            dy, g, sub, list(gate_shape), need_g, need_s, self.threads, self.grid_cap
        )
        return dg if need_g else None, ds if need_s else None
