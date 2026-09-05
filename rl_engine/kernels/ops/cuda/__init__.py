# SPDX-License-Identifier: Apache-2.0
"""Load CUDA operator families on demand so optional backends remain optional."""

from importlib import import_module

__all__ = ["activation", "attention", "loss", "matmul", "modulation", "norm"]


def __getattr__(name):
    if name not in __all__:
        raise AttributeError(name)
    module = import_module(f"{__name__}.{name}")
    globals()[name] = module
    return module
