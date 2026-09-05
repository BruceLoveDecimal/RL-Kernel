# SPDX-License-Identifier: Apache-2.0
import itertools

import pytest
import torch

from rl_engine.kernels.adaln_gate_residual import adaln_gate_residual
from rl_engine.testing.adaln_gate_residual_reference import (
    bit_check,
    reference_backward_fp32,
    reference_forward_fp32,
)


@pytest.fixture(params=["pytorch", "cuda", "triton"])
def backend_device(request):
    backend = request.param
    if backend != "pytorch" and not torch.cuda.is_available():
        pytest.skip("NVIDIA GPU required; strict CLI rejects missing GPU")
    return backend, "cpu" if backend == "pytorch" else "cuda"


def inputs(shape, mode, dtype, mixed=False):
    b, seq, d = shape
    rng = torch.Generator().manual_seed(386)
    x = torch.randn(shape, generator=rng).to(dtype)
    s = torch.randn(shape, generator=rng).to(dtype)
    gs = {"bd": (b, d), "b1d": (b, 1, d), "bsd": shape}[mode]
    g = torch.randn(gs, generator=rng).to(torch.float32 if mixed else dtype)
    dy = torch.randn(shape, generator=rng).to(dtype)
    return x, g, s, dy


def assert_bits(actual, expected):
    report = bit_check(actual, expected)
    assert report["passed"], report


@pytest.mark.parametrize(
    "shape",
    [
        (2, 0, 33),
        (1, 1, 1),
        (2, 31, 31),
        (2, 255, 32),
        (1, 256, 33),
        (2, 257, 127),
        (1, 513, 129),
        (1, 2, 3071),
        (1, 2, 3072),
        (1, 2, 3073),
    ],
)
@pytest.mark.parametrize("mode", ["bd", "b1d", "bsd"])
@pytest.mark.parametrize(
    "dtype,mixed", [(torch.float32, False), (torch.bfloat16, False), (torch.bfloat16, True)]
)
def test_oracle(backend_device, shape, mode, dtype, mixed):
    backend, device = backend_device
    x, g, s, dy = inputs(shape, mode, dtype, mixed)
    expected = reference_forward_fp32(x, g, s).to(dtype)
    dx, dg, ds = reference_backward_fp32(dy, g, s)
    tensors = [t.to(device).requires_grad_() for t in (x, g, s)]
    trace = {}
    result = adaln_gate_residual(*tensors, backend=backend, trace=trace)
    grads = torch.autograd.grad(result, tensors, dy.to(device))
    assert_bits(result, expected)
    for actual, gold, original in zip(grads, (dx, dg, ds), (x, g, s), strict=False):
        assert_bits(actual, gold.to(original.dtype))
    assert trace["forward_backend"] == trace["backward_backend"] == backend
    assert trace["kernel_fingerprint"]


@pytest.mark.parametrize("needs", list(itertools.product([False, True], repeat=3)))
def test_partial_gradients(backend_device, needs):
    backend, device = backend_device
    x, g, s, dy = inputs((2, 257, 33), "bd", torch.bfloat16, True)
    gold = reference_backward_fp32(dy, g, s)
    tensors = [t.to(device).requires_grad_(n) for t, n in zip((x, g, s), needs, strict=False)]
    out = adaln_gate_residual(*tensors, backend=backend)
    if not any(needs):
        assert not out.requires_grad
        return
    grads = torch.autograd.grad(
        out, [t for t, n in zip(tensors, needs, strict=False) if n], dy.to(device)
    )
    for actual, expected in zip(
        grads,
        [v.to(t.dtype) for v, t, n in zip(gold, tensors, needs, strict=False) if n],
        strict=False,
    ):
        assert_bits(actual, expected)


def test_noncontiguous_and_versions(backend_device):
    backend, device = backend_device
    rng = torch.Generator().manual_seed(42)
    x = torch.randn((2, 33, 257), generator=rng).to(device).transpose(1, 2).requires_grad_()
    s = torch.randn((2, 257, 34), generator=rng).to(device)[:, :, 1:].requires_grad_()
    g = torch.randn((2, 99), generator=rng).to(device).chunk(3, dim=-1)[2].requires_grad_()
    dy = torch.randn((2, 33, 257), generator=rng).to(device).transpose(1, 2)
    versions = [t._version for t in (x, g, s)]
    clones = [t.detach().cpu().clone() for t in (x, g, s)]
    trace = {}
    out = adaln_gate_residual(x, g, s, backend=backend, trace=trace)
    grads = torch.autograd.grad(out, (x, g, s), dy)
    assert out.data_ptr() not in [t.data_ptr() for t in (x, g, s)]
    assert versions == [t._version for t in (x, g, s)]
    assert trace["layout_copy"] == [True, True, True]
    for got, want in zip(
        grads, reference_backward_fp32(dy.cpu(), clones[1], clones[2]), strict=False
    ):
        assert_bits(got, want)


def test_adversarial_rounding(backend_device):
    backend, device = backend_device
    # (1+2^-23)*(1-2^-23)-1: separate FP32 gives zero; FMA gives -2^-46.
    x = torch.tensor([-1.0, 0.0, -0.0, 0.0, 0.0]).reshape(1, 1, 5)
    g = torch.tensor([1 + 2**-23, 1.0, -1.0, 1.0, 1.0]).reshape(1, 1, 5)
    s = torch.tensor([1 - 2**-23, 2**-149, 0.0, -(2**-149), 2**-126]).reshape(1, 1, 5)
    out = adaln_gate_residual(*[t.to(device) for t in (x, g, s)], backend=backend)
    assert_bits(out, reference_forward_fp32(x, g, s))
    # BF16 tie / cancellation corpus is derived from exactly representable inputs.
    for shift in [2**-8, 3 * 2**-8, -(2**-8)]:
        xb = torch.ones((1, 1, 5), dtype=torch.bfloat16)
        gb = torch.ones_like(xb)
        sb = torch.full_like(xb, shift)
        out = adaln_gate_residual(*[t.to(device) for t in (xb, gb, sb)], backend=backend)
        assert_bits(out, reference_forward_fp32(xb, gb, sb).bfloat16())


def test_zero_gate_vjp(backend_device):
    backend, device = backend_device
    x = torch.ones((1, 257, 3), device=device, requires_grad=True)
    g = torch.zeros((1, 3), device=device, requires_grad=True)
    s = x.detach().clone().requires_grad_()
    out = adaln_gate_residual(x, g, s, backend=backend)
    dx, dg, ds = torch.autograd.grad(out, (x, g, s), torch.ones_like(out))
    assert_bits(out, x.detach())
    assert_bits(dx, torch.ones_like(x))
    assert_bits(ds, torch.zeros_like(s))
    assert_bits(dg, torch.full_like(g, 257))


def test_oracle_hand_computed():
    dy = torch.ones(1, 4, 1)
    g = torch.ones(1, 1)
    s = torch.tensor([2**24, 1.0, -(2**24), 1.0]).reshape(1, 4, 1)
    # Adjacent pairs: 2^24 + (-2^24 + 1) = 1 (not a generic sequential sum).
    assert_bits(reference_backward_fp32(dy, g, s)[1], torch.ones(1, 1))
    assert not bit_check(torch.tensor([0.0]), torch.tensor([-0.0]))["passed"]


def test_lora_two_residuals():
    torch.manual_seed(386)
    x = torch.randn(2, 5, 8, requires_grad=True)
    base = torch.randn(8, 8)  # deliberately frozen
    a = torch.randn(8, 2, requires_grad=True)
    b = torch.randn(2, 8, requires_grad=True)
    g = torch.randn(2, 1, 8, requires_grad=True)
    h = x
    for _ in range(2):
        sub = h @ base + (h @ a) @ b
        h = adaln_gate_residual(h, g, sub, backend="pytorch")
    h.square().mean().backward()
    assert base.grad is None
    for t in (a, b):
        assert torch.isfinite(t.grad).all() and torch.count_nonzero(t.grad) > 0


def test_math_vjp_float64_sanity():
    x, g, s, dy = inputs((1, 4, 3), "bd", torch.float32)
    tensors = [t.double().requires_grad_() for t in (x, g, s)]
    y = tensors[0] + tensors[1][:, None] * tensors[2]
    gold = torch.autograd.grad(y, tensors, dy.double())
    ref = reference_backward_fp32(dy, g, s)
    for a, b in zip(ref, gold, strict=False):
        torch.testing.assert_close(a.double(), b, rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize("backend", ["pytorch", "cuda", "triton"])
def test_nonfinite_propagation(backend):
    if backend != "pytorch" and not torch.cuda.is_available():
        pytest.skip("GPU required")
    device = "cpu" if backend == "pytorch" else "cuda"
    x = torch.tensor([float("inf"), -float("inf"), float("nan"), 1.0]).reshape(1, 1, 4)
    s = torch.tensor([1.0, 1.0, 1.0, float("inf")]).reshape(1, 1, 4)
    g = torch.ones_like(x)
    expected = reference_forward_fp32(x, g, s)
    got = adaln_gate_residual(*[t.to(device) for t in (x, g, s)], backend=backend).cpu()
    assert torch.equal(torch.isnan(got), torch.isnan(expected))
    torch.testing.assert_close(got, expected, rtol=0, atol=0, equal_nan=True)


def test_second_derivatives_rejected():
    x, g, s = [torch.ones(1, 2, 3, requires_grad=True) for _ in range(3)]
    y = adaln_gate_residual(x, g, s, backend="pytorch")
    dg = torch.autograd.grad(y, g, torch.ones_like(y), create_graph=True)[0]
    with pytest.raises(RuntimeError):
        torch.autograd.grad(dg.sum(), s)


def test_saved_input_version_check():
    x, g, s = [torch.ones(1, 2, 3, requires_grad=True) for _ in range(3)]
    y = adaln_gate_residual(x, g, s, backend="pytorch")
    with torch.no_grad():
        g.add_(1)
    with pytest.raises(RuntimeError, match="modified by an inplace"):
        y.sum().backward()


def test_fixed_multi_path_sum():
    # Test the prescribed external merge explicitly, independent of an engine's graph schedule.
    x, g, s, dy = inputs((1, 257, 33), "b1d", torch.bfloat16)
    direct, _, ds = reference_backward_fp32(dy, g, s)
    upstream = torch.full_like(ds, 0.125)
    tensors = [t.requires_grad_() for t in (x, g, s)]
    y = adaln_gate_residual(*tensors, backend="pytorch")
    dx, _, local_ds = torch.autograd.grad(y, tensors, dy)
    # Local ds is already rounded at the operator boundary; match this boundary in the oracle.
    expected = (direct + ds.bfloat16().float() * upstream).bfloat16()
    merged = (dx.float() + local_ds.float() * upstream).bfloat16()
    assert_bits(merged, expected)
