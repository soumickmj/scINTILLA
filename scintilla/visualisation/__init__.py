"""Visualisation sub-package."""

import matplotlib
matplotlib.use("Agg")

from scintilla.visualisation.clustering_plots import (
    ari_benchmark_plot,
    dendrogram_plot,
    cophenetic_vs_original_scatter,
)
from scintilla.visualisation.classification_plots import (
    benchmark_bar_chart,
    confusion_matrix_heatmap,
)
from scintilla.visualisation.pca_plots import (
    pca_2d_plot,
    pca_3d_multiview,
    cumulative_variance_plot,
)

__all__ = [
    "ari_benchmark_plot",
    "dendrogram_plot",
    "cophenetic_vs_original_scatter",
    "benchmark_bar_chart",
    "confusion_matrix_heatmap",
    "pca_2d_plot",
    "pca_3d_multiview",
    "cumulative_variance_plot",
]
