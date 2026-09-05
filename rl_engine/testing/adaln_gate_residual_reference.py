# SPDX-License-Identifier: Apache-2.0
"""CPU-only oracle; explicit rounding and reduction order, independent of kernels."""

import torch


def _cpu(*inputs):
    if any(t.device.type != "cpu" for t in inputs):
        raise ValueError("the reference oracle requires CPU tensors")


def reference_forward_fp32(x, gate, sublayer_out):
    _cpu(x, gate, sublayer_out)
    g = gate.unsqueeze(1) if gate.ndim == 2 else gate
    product = torch.mul(g.float(), sublayer_out.float())
    return torch.add(x.float(), product)


def reference_backward_fp32(dy, gate, sublayer_out):
    _cpu(dy, gate, sublayer_out)
    g = gate.unsqueeze(1) if gate.ndim == 2 else gate
    ds = torch.mul(dy.float(), g.float())
    broadcast = gate.ndim == 2 or gate.shape[1] == 1
    if not broadcast:
        dg = torch.mul(dy.float(), sublayer_out.float())
    else:
        b, seq, d = dy.shape
        dg = torch.zeros((b, d), dtype=torch.float32)
        for start in range(0, seq, 256):
            leaves = torch.zeros((b, 256, d), dtype=torch.float32)
            count = min(256, seq - start)
            leaves[:, :count] = torch.mul(
                dy[:, start : start + count].float(), sublayer_out[:, start : start + count].float()
            )
            for _ in range(8):
                leaves = torch.add(leaves[:, 0::2], leaves[:, 1::2])
            dg = torch.add(dg, leaves[:, 0])
        dg = dg.reshape(gate.shape)
    return dy.float().clone(), dg, ds


def bit_check(actual, expected):
    """Report bit mismatches including signed zero, without floating equality shortcuts."""
    a, e = actual.detach().cpu().contiguous(), expected.detach().cpu().contiguous()
    if a.shape != e.shape or a.dtype != e.dtype:
        return {"passed": False, "reason": "shape/dtype mismatch"}
    integer = torch.int32 if a.dtype == torch.float32 else torch.int16
    ai, ei = a.view(integer).reshape(-1), e.view(integer).reshape(-1)
    bad = ai != ei
    indices = bad.nonzero().flatten()
    result = {"passed": not bool(indices.numel()), "mismatch_count": indices.numel()}
    if indices.numel():
        i = int(indices[0])
        mask = (1 << (a.element_size() * 8)) - 1
        result.update(
            first_flat_index=i,
            actual_bits=hex(int(ai[i]) & mask),
            expected_bits=hex(int(ei[i]) & mask),
            max_abs_error=float((a.float() - e.float()).abs().max()),
        )
    return result
