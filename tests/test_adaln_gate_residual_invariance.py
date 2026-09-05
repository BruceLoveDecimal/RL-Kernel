# SPDX-License-Identifier: Apache-2.0
import pytest
import torch

from rl_engine.kernels.adaln_gate_residual import adaln_gate_residual, load_backend
from rl_engine.testing.adaln_gate_residual_reference import bit_check


def run(op, values):
    x, g, s, dy = values
    ts = [t.detach().clone().requires_grad_() for t in (x, g, s)]
    out = op(*ts)
    return [out.detach(), *torch.autograd.grad(out, ts, dy)]


def same(a, b):
    for left, right in zip(a, b, strict=False):
        report = bit_check(left, right)
        assert report["passed"], report


@pytest.mark.parametrize("backend", ["pytorch", "cuda", "triton"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_batch_launch_repeat(backend, dtype):
    if backend != "pytorch" and not torch.cuda.is_available():
        pytest.skip("GPU required")
    device = "cpu" if backend == "pytorch" else "cuda"
    torch.manual_seed(386)
    values = [
        torch.randn(7, 257, 33),
        torch.randn(7, 1, 33),
        torch.randn(7, 257, 33),
        torch.randn(7, 257, 33),
    ]
    values = [t.to(device=device, dtype=dtype) for t in values]
    op = load_backend(backend)
    target = [t[3:4] for t in values]
    gold = run(op, target)
    for size in (2, 4, 7):
        for position in (0, size - 1):
            batch = [t[:size].clone() for t in values]
            for tensor, one in zip(batch, target, strict=False):
                tensor[position : position + 1] = one
            result = run(op, batch)
            same([t[position : position + 1] for t in result], gold)
    full = run(op, values)
    chunks = [run(op, [t[i : i + 2] for t in values]) for i in range(0, 7, 2)]
    same([torch.cat([chunk[j] for chunk in chunks]) for j in range(4)], full)
    for _ in range(100):
        same(run(op, target), gold)
    if backend == "cuda":
        other = type(op)(threads=128, grid_cap=1)
        same(run(other, values), full)
    if backend == "triton":
        other = type(op)(num_warps=8, block=128, feature=8)
        same(run(other, values), full)
    if device == "cuda":
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            got = run(op, values)
        torch.cuda.current_stream().wait_stream(stream)
        same(got, full)


@pytest.mark.parametrize("backend", ["pytorch", "cuda", "triton"])
def test_token_shape_invariance(backend):
    if backend != "pytorch" and not torch.cuda.is_available():
        pytest.skip("GPU required")
    device = "cpu" if backend == "pytorch" else "cuda"
    torch.manual_seed(6)
    x, s, dy = [torch.randn(1, 513, 33, device=device) for _ in range(3)]
    g = torch.randn(1, 1, 33, device=device)
    op = load_backend(backend)
    gold = run(op, [x, g, s, dy])
    # Only pointwise outputs may be compared after changing active token count.
    got = run(op, [x[:, 255:257], g, s[:, 255:257], dy[:, 255:257]])
    same([got[i] for i in (0, 1, 3)], [gold[i][:, 255:257] for i in (0, 1, 3)])


def test_expanded_input_and_alias_local_vjp():
    x = torch.randn(1, 3, 4, requires_grad=True)
    g = torch.randn(1, 1, 4, requires_grad=True)
    # Expanded gate is a separate input slot: autograd owns any upstream reduction.
    expanded = g.expand(1, 3, 4)
    y = adaln_gate_residual(x, expanded, x, backend="pytorch")
    dx = torch.autograd.grad(y, x, torch.ones_like(y))[0]
    same([dx], [torch.ones_like(x) + expanded])
