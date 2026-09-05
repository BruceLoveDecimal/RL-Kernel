# SPDX-License-Identifier: Apache-2.0
"""Explicit FP32 PyTorch fallback with a frozen broadcast-gradient tree."""

import torch

from rl_engine.kernels.adaln_gate_residual import BackendUnavailable, ResidualOp


class NativeAdaLNGateResidualOp(ResidualOp):
    backend = "pytorch"

    def check_available(self, device):
        if torch.device(device).type not in ("cpu", "cuda"):
            raise BackendUnavailable("v1 PyTorch path is validated only on CPU/CUDA devices")

        if torch.device(device).type == "cpu":
            probe = torch.tensor([torch.finfo(torch.float32).tiny], dtype=torch.float32) * 0.5
            if probe.view(torch.int32).item() != 0x00400000:
                raise BackendUnavailable("CPU FTZ must be disabled for the strict contract")

    def forward_raw(self, x, gate, sublayer_out):
        g = gate.unsqueeze(1) if gate.ndim == 2 else gate
        p = g.float() * sublayer_out.float()
        return (x.float() + p).to(x.dtype)

    def backward_raw(self, dy, gate, s, gate_shape, gate_dtype, broadcast, need_g, need_s):
        self.check_available(dy.device)
        ds = dg = None
        if need_s:
            g = gate.unsqueeze(1) if gate.ndim == 2 else gate
            ds = (dy.float() * g.float()).to(dy.dtype)
        if need_g:
            if broadcast:
                b, seq, d = dy.shape
                acc = torch.zeros((b, d), device=dy.device, dtype=torch.float32)
                for start in range(0, seq, 256):
                    stop = min(start + 256, seq)
                    values = dy[:, start:stop].float() * s[:, start:stop].float()
                    values = torch.nn.functional.pad(values, (0, 0, 0, 256 - (stop - start)))
                    width = 256
                    while width > 1:
                        values = values.reshape(b, width // 2, 2, d)
                        values = values[:, :, 0] + values[:, :, 1]
                        width //= 2
                    acc = acc + values[:, 0]
                dg = acc.reshape(gate_shape).to(gate_dtype)
            else:
                dg = (dy.float() * s.float()).to(gate_dtype)
        return dg, ds
