"""Benchmarks: compare methods at each stage and measure cost.

Each ``benchmark_*`` function returns data (a results table, possibly with a leaderboard dict);
a method that fails stays in the table with ``status="failed"`` and a reason.  Draw the tables
with :mod:`scintilla.pl`.
"""

from scintilla.batch_correction.benchmark import benchmark_batch_correction
from scintilla.benchmarking import (
    BenchmarkReport,
    BenchmarkResult,
    estimate_benchmark_time,
    pairwise_method_comparison,
    print_time_budget,
    profile_method,
    scalability_sweep,
    seed_stability_test,
)
from scintilla.classification.benchmark import benchmark_models_comprehensive as benchmark_classifiers
from scintilla.classification.benchmark import compare_classifiers
from scintilla.clustering.benchmark import benchmark_clustering_methods as benchmark_clustering
from scintilla.differential_expression.benchmark import benchmark_de_methods as benchmark_de
from scintilla.dimensionality_reduction.benchmark import benchmark_embeddings
from scintilla.feature_selection.benchmark import benchmark_feature_selection, hvg_sensitivity_analysis
from scintilla.preprocessing.benchmark import benchmark_transformations

__all__ = [
    "benchmark_transformations", "benchmark_clustering", "benchmark_classifiers",
    "benchmark_feature_selection", "hvg_sensitivity_analysis", "benchmark_embeddings",
    "benchmark_de", "benchmark_batch_correction", "compare_classifiers",
    "estimate_benchmark_time", "print_time_budget", "profile_method", "scalability_sweep",
    "seed_stability_test", "pairwise_method_comparison", "BenchmarkReport", "BenchmarkResult",
]
