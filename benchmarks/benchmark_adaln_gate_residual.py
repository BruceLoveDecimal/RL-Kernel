# SPDX-License-Identifier: Apache-2.0
"""Public API GPU latency including layout normalization, excluding compilation."""

import argparse
import json
import statistics
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from rl_engine.kernels.adaln_gate_residual import adaln_gate_residual, load_backend  # noqa: E402
from scripts.validate_adaln_gate_residual import environment  # noqa: E402


def measure(fn, warmup=20, repeats=100):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    allocated = torch.cuda.memory_allocated()
    times = []
    for _ in range(repeats):
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        end.synchronize()
        times.append(start.elapsed_time(end))
    return {
        "median_ms": statistics.median(times),
        "p95_ms": sorted(times)[int(0.95 * len(times))],
        "peak_extra_bytes": torch.cuda.max_memory_allocated() - allocated,
        "warmup": warmup,
        "repeats": repeats,
    }


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--backend", default="all", choices=["all", "pytorch", "cuda", "triton"])
    parser.add_argument("--suite", default="qwen-image", choices=["smoke", "qwen-image"])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("benchmark requires CUDA")
    backends = ["pytorch", "triton", "cuda", "eager"] if args.backend == "all" else [args.backend]
    shapes = (
        [(1, 257, 33)]
        if args.suite == "smoke"
        else [(b, s, 3072) for b in (1, 2, 4) for s in (77, 4096, 6889, 6032)]
    )
    results = []
    for shape in shapes:
        for dtype in (torch.float32, torch.bfloat16):
            for mode in ("broadcast", "elementwise"):
                torch.manual_seed(386)
                x, s = [
                    torch.randn(shape, device="cuda", dtype=dtype, requires_grad=True)
                    for _ in range(2)
                ]
                gshape = (shape[0], 1, shape[2]) if mode == "broadcast" else shape
                g = torch.randn(gshape, device="cuda", dtype=dtype, requires_grad=True)
                dy = torch.randn_like(x)
                for backend in backends:

                    def forward(x=x, g=g, s=s, backend=backend):
                        if backend == "eager":
                            return x + g * s
                        return adaln_gate_residual(x, g, s, backend=backend)

                    y = forward()

                    def backward(y=y, x=x, g=g, s=s, dy=dy):
                        return torch.autograd.grad(y, (x, g, s), dy, retain_graph=True)

                    def both(forward=forward, x=x, g=g, s=s, dy=dy):
                        return torch.autograd.grad(forward(), (x, g, s), dy)

                    result = {
                        "backend": backend,
                        "shape": shape,
                        "dtype": str(dtype),
                        "gate": mode,
                        "strict_contract": backend != "eager",
                        "scope": "public_api_including_normalization",
                        "forward": measure(forward),
                        "backward": measure(backward),
                        "forward_backward": measure(both),
                    }
                    if backend != "eager":
                        op = load_backend(backend)
                        op.check_available(x.device)

                        def raw_forward(op=op, x=x, g=g, s=s):
                            return op.forward_raw(x, g, s)

                        def raw_backward(op=op, dy=dy, g=g, s=s, mode=mode):
                            return op.backward_raw(
                                dy, g, s, g.shape, g.dtype, mode == "broadcast", True, True
                            )

                        result["forward_raw"] = measure(raw_forward)
                        result["backward_raw"] = measure(raw_backward)
                        ebytes, gbytes = x.numel() * x.element_size(), g.numel() * g.element_size()
                        partial = 4 * shape[0] * ((shape[1] + 255) // 256) * shape[2]
                        result["logical_bytes"] = {
                            "forward_minimum": 3 * ebytes + gbytes,
                            "backward": (
                                4 * ebytes + 2 * gbytes + 2 * partial
                                if mode == "broadcast"
                                else 3 * ebytes + 2 * gbytes
                            ),
                            "note": "ideal gate reuse; excludes allocator/cache transactions",
                        }
                        result["forward_effective_gb_s"] = result["logical_bytes"][
                            "forward_minimum"
                        ] / (result["forward_raw"]["median_ms"] * 1e6)
                    results.append(result)
                    print(json.dumps(result), flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps({"environment": environment(), "results": results}, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
