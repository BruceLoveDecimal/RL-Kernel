# SPDX-License-Identifier: Apache-2.0
"""Bit-level CPU/GPU acceptance runner. Missing GPU never passes --require-gpu."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import tempfile
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from rl_engine.kernels.adaln_gate_residual import adaln_gate_residual  # noqa: E402
from rl_engine.testing.adaln_gate_residual_reference import (  # noqa: E402
    bit_check,
    reference_backward_fp32,
    reference_forward_fp32,
)


def environment():
    def command(*args):
        try:
            return subprocess.check_output(
                args, cwd=ROOT, text=True, stderr=subprocess.STDOUT
            ).strip()
        except (OSError, subprocess.CalledProcessError) as exc:
            return str(exc)

    return {
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "git_sha": command("git", "rev-parse", "HEAD"),
        "git_status": command("git", "status", "--short"),
        "gpu": command("nvidia-smi"),
        "nvcc": command("nvcc", "--version"),
        "compiler": command("c++", "--version"),
    }


def make_case(b, seq, d, mode, dtype, mixed):
    rng = torch.Generator().manual_seed(386)
    shape = (b, seq, d)
    gs = {"bd": (b, d), "b1d": (b, 1, d), "bsd": shape}[mode]
    return [
        torch.randn(shape, generator=rng).to(dtype),
        torch.randn(gs, generator=rng).to(torch.float32 if mixed else dtype),
        torch.randn(shape, generator=rng).to(dtype),
        torch.randn(shape, generator=rng).to(dtype),
    ]


def check_case(backend, b, seq, d, mode, dtype, mixed, device):
    x, g, s, dy = make_case(b, seq, d, mode, dtype, mixed)
    data_hash = hashlib.sha256()
    for tensor in (x, g, s, dy):
        data_hash.update(tensor.contiguous().view(torch.uint8).numpy().tobytes())
    expected = [reference_forward_fp32(x, g, s).to(dtype)]
    expected += [
        v.to(t.dtype) for v, t in zip(reference_backward_fp32(dy, g, s), (x, g, s), strict=False)
    ]
    tensors = [t.to(device).requires_grad_() for t in (x, g, s)]
    trace = {}
    y = adaln_gate_residual(*tensors, backend=backend, trace=trace)
    actual = [y, *torch.autograd.grad(y, tensors, dy.to(device))]
    checks = {
        name: bit_check(a, e)
        for name, a, e in zip(("y", "dx", "dgate", "ds"), actual, expected, strict=False)
    }
    return {
        "backend": backend,
        "shape": [b, seq, d],
        "mode": mode,
        "dtype": str(dtype),
        "mixed": mixed,
        "seed": 386,
        "input_hash": data_hash.hexdigest(),
        "trace": trace,
        "checks": checks,
        "passed": all(v["passed"] for v in checks.values()),
    }


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--backend", choices=["all", "pytorch", "cuda", "triton"], default="all")
    parser.add_argument("--suite", choices=["smoke", "qwen-image", "strict"], default="strict")
    parser.add_argument("--require-gpu", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(min(8, os.cpu_count() or 1))
    report = {"environment": environment(), "suite": args.suite, "cases": [], "passed": False}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    try:
        gpu = torch.cuda.is_available() and torch.version.hip is None
        if args.require_gpu and not gpu:
            raise RuntimeError("--require-gpu: NVIDIA CUDA device unavailable")
        backends = ["pytorch", "cuda", "triton"] if args.backend == "all" else [args.backend]
        if not gpu and any(b != "pytorch" for b in backends):
            raise RuntimeError(
                "GPU backends requested but unavailable; use --backend pytorch for CPU"
            )
        shapes = (
            [(1, 257, 33)]
            if args.suite == "smoke"
            else [(b, seq, 3072) for b in (1, 2, 4) for seq in (4096, 6889, 6032, 77, 256, 512)]
        )
        for shape in shapes:
            for dtype, mixed in (
                (torch.float32, False),
                (torch.bfloat16, False),
                (torch.bfloat16, True),
            ):
                for mode in ("bd", "b1d", "bsd"):
                    for backend in backends:
                        case = check_case(
                            backend, *shape, mode, dtype, mixed, "cuda" if gpu else "cpu"
                        )
                        report["cases"].append(case)
                        print(
                            json.dumps(
                                {k: case[k] for k in ("backend", "shape", "mode", "passed")}
                            ),
                            flush=True,
                        )
                        if not case["passed"]:
                            raise AssertionError("bit mismatch; see report")
        if args.suite == "strict":
            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "pytest",
                    "tests/test_adaln_gate_residual.py",
                    "tests/test_adaln_gate_residual_invariance.py",
                    "tests/test_adaln_gate_residual_dispatch.py",
                    "-q",
                    "-rs",
                ],
                cwd=ROOT,
                capture_output=True,
                text=True,
            )
            report["pytest"] = {
                "returncode": result.returncode,
                "output": result.stdout + result.stderr,
            }
            if result.returncode:
                raise AssertionError("focused acceptance tests failed")
            report["cold_processes"] = []
            for repetition in range(3):
                with tempfile.TemporaryDirectory(prefix="adaln-cold-") as cache:
                    output = args.output.parent / f"adaln-cold-{repetition}.json"
                    command = [
                        sys.executable,
                        str(Path(__file__).resolve()),
                        "--backend",
                        args.backend,
                        "--suite",
                        "smoke",
                        "--output",
                        str(output.resolve()),
                    ]
                    if args.require_gpu:
                        command.append("--require-gpu")
                    result = subprocess.run(
                        command,
                        cwd=ROOT,
                        capture_output=True,
                        text=True,
                        env={**os.environ, "TRITON_CACHE_DIR": cache},
                    )
                    report["cold_processes"].append(
                        {"returncode": result.returncode, "report": str(output)}
                    )
                    if result.returncode:
                        raise AssertionError("cold process/JIT acceptance failed")
        report["passed"] = True
    except Exception as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
