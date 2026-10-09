"""Import the optional ``louvain`` package on setuptools >= 82.

``louvain`` 0.8.2 (the last release) does ``from pkg_resources import get_distribution`` at import
time, and setuptools removed ``pkg_resources`` in version 82. Pinning ``setuptools<82`` would keep
a build tool with known advisories in every environment, so instead a two-name stand-in for
``pkg_resources`` is provided *only while* ``louvain`` is being imported and removed again afterwards.
"""

from __future__ import annotations

import importlib
import importlib.metadata
import sys
import types
from contextlib import contextmanager
from types import ModuleType
from typing import Iterator


def _stand_in() -> ModuleType:
    module = types.ModuleType("pkg_resources")
    module.DistributionNotFound = importlib.metadata.PackageNotFoundError  # type: ignore[attr-defined]

    def get_distribution(name: str):
        return importlib.metadata.distribution(name)

    module.get_distribution = get_distribution  # type: ignore[attr-defined]
    return module


@contextmanager
def _pkg_resources_if_missing() -> Iterator[None]:
    try:
        importlib.import_module("pkg_resources")
    except ImportError:
        sys.modules["pkg_resources"] = _stand_in()
        try:
            yield
        finally:
            sys.modules.pop("pkg_resources", None)
    else:
        yield


def import_louvain() -> ModuleType:
    """Import and return the ``louvain`` module, raising ``ImportError`` if it is not installed."""
    if "louvain" in sys.modules:
        return sys.modules["louvain"]
    with _pkg_resources_if_missing():
        return importlib.import_module("louvain")
