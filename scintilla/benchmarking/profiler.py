"""Profiling utilities for benchmarking."""

from __future__ import annotations

import tracemalloc
import time
from collections import namedtuple
from functools import wraps
from typing import Callable

BenchmarkResult = namedtuple("BenchmarkResult", ["result", "elapsed_seconds", "peak_memory_mb"])


def profile_method(func: Callable) -> Callable:
    """Decorator that captures wall-clock time and peak memory usage.

    Parameters
    ----------
    func:
        Function to profile.

    Returns
    -------
    Wrapped function returning BenchmarkResult(result, elapsed_seconds, peak_memory_mb).
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        tracemalloc.start()
        t0 = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed = time.perf_counter() - t0
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        peak_mb = peak / 1024 / 1024
        return BenchmarkResult(result=result, elapsed_seconds=elapsed, peak_memory_mb=peak_mb)

    return wrapper
