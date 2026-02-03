"""qdre package

This package lazily exposes its primary submodules as attributes on
`qdre` to reduce import time and avoid importing heavy dependencies
unless they're actually used.

Example
-------
>>> from qdre import plotting  # submodules are imported lazily when accessed
>>> plotting.some_function()
"""

__version__ = "0.0.0"

# Lazy import map: attribute name -> full module path
from importlib import import_module
from typing import Any

_lazy_map = {
    "preprocessing": "qdre.preprocessing",
    "models": "qdre.models",
    "train": "qdre.train",
    "plotting": "qdre.plotting",
    "metrics": "qdre.metrics",
}


def __getattr__(name: str) -> Any:
    """Lazy-load a submodule when it's accessed as an attribute on the package."""
    if name in _lazy_map:
        mod = import_module(_lazy_map[name])
        globals()[name] = mod
        return mod
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(list(globals().keys()) + list(_lazy_map.keys()))


__all__ = list(_lazy_map.keys()) + ["__version__"]
