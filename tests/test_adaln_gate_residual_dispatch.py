# SPDX-License-Identifier: Apache-2.0
import importlib

import pytest
import torch

api = importlib.import_module("rl_engine.kernels.adaln_gate_residual")


def test_cpu_registry():
    from rl_engine.kernels.registry import KernelRegistry

    op = KernelRegistry().get_op("adaln_gate_residual", device="cpu")
    t = torch.ones(1, 2, 3)
    trace = {}
    assert torch.equal(op(t, t, t, trace=trace), t * 2)
    assert trace["forward_backend"] == "pytorch"
    assert trace["fallback_reasons"] == []


@pytest.mark.parametrize("dtype", [torch.float16, torch.float64, torch.int32])
def test_reject_dtype(dtype):
    t = torch.ones(1, 2, 3, dtype=dtype)
    with pytest.raises(TypeError):
        api.adaln_gate_residual(t, t, t)


@pytest.mark.parametrize("shape", [(), (3,), (1, 2, 1), (2, 3), (1, 3, 3)])
def test_reject_gate_shape(shape):
    t = torch.ones(1, 2, 3)
    with pytest.raises(ValueError):
        api.adaln_gate_residual(t, torch.ones(shape), t)


def test_explicit_backend_unavailable(monkeypatch):
    from rl_engine.kernels.ops import base

    monkeypatch.setattr(base, "_C", None)
    t = torch.ones(1, 2, 3)
    with pytest.raises(api.BackendUnavailable):
        api.adaln_gate_residual(t, t, t, backend="cuda")
    with pytest.raises(ValueError):
        api.adaln_gate_residual(t, t, t, backend="misspelled")


def test_missing_backward_symbol(monkeypatch):
    from rl_engine.kernels.ops import base
    from rl_engine.kernels.ops.cuda.modulation.adaln_gate_residual import CudaAdaLNGateResidualOp

    class Incomplete:
        adaln_gate_residual_forward = True

    monkeypatch.setattr(base, "_C", Incomplete())
    with pytest.raises(api.BackendUnavailable):
        CudaAdaLNGateResidualOp()


def test_no_retry_after_execution_failure(monkeypatch):
    class Broken:
        def check_available(self, device):
            pass

        def __call__(self, *args, **kwargs):
            raise RuntimeError("kernel fault")

    monkeypatch.setattr(api, "load_backend", lambda name: Broken())
    t = torch.ones(1, 2, 3)
    with pytest.raises(RuntimeError, match="kernel fault"):
        api.adaln_gate_residual(t, t, t)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU required")
def test_auto_fallback_is_observable(monkeypatch):
    real_load = api.load_backend

    def only_native(name):
        if name != "pytorch":
            raise api.BackendUnavailable(f"test: missing {name}")
        return real_load(name)

    monkeypatch.setattr(api, "load_backend", only_native)
    t = torch.ones(1, 2, 3, device="cuda")
    trace = {}
    out = api.adaln_gate_residual(t, t, t, trace=trace)
    assert torch.equal(out, t * 2)
    assert trace["forward_backend"] == "pytorch"
    assert len(trace["fallback_reasons"]) == 2


def test_cpu_flush_to_zero_rejected():
    # set_flush_denormal returns False on unsupported CPU architectures.
    if not torch.set_flush_denormal(True):
        pytest.skip("CPU cannot configure FTZ")
    try:
        t = torch.ones(1, 2, 3)
        with pytest.raises(api.BackendUnavailable, match="FTZ"):
            api.adaln_gate_residual(t, t, t, backend="pytorch")
    finally:
        torch.set_flush_denormal(False)
