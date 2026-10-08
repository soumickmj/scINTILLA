"""Feature selection sub-package."""

from scintilla.feature_selection.pca_loadings import (
    extract_top_genes_per_pc,
    build_reduced_dataset,
    validate_reduced_set,
)
from scintilla.feature_selection.hvg import select_hvg
from scintilla.feature_selection.mutual_information import mi_feature_selection
from scintilla.feature_selection.boruta import boruta_selection
from scintilla.feature_selection.mrmr import mrmr_selection
from scintilla.feature_selection.benchmark import benchmark_feature_selection

__all__ = [
    "extract_top_genes_per_pc",
    "build_reduced_dataset",
    "validate_reduced_set",
    "select_hvg",
    "mi_feature_selection",
    "boruta_selection",
    "mrmr_selection",
    "benchmark_feature_selection",
]
