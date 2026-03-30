"""Benchmarking sub-package."""

from scintilla.benchmarking.profiler import profile_method, BenchmarkResult
from scintilla.benchmarking.scalability import scalability_sweep
from scintilla.benchmarking.reporter import BenchmarkReport
from scintilla.benchmarking.reproducibility import seed_stability_test
from scintilla.benchmarking.comparison import pairwise_method_comparison
from scintilla.benchmarking.time_estimator import estimate_benchmark_time, print_time_budget

__all__ = [
    "profile_method",
    "BenchmarkResult",
    "scalability_sweep",
    "BenchmarkReport",
    "seed_stability_test",
    "pairwise_method_comparison",
    "estimate_benchmark_time",
    "print_time_budget",
]
