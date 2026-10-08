"""Preprocessing sub-package."""

from scintilla.preprocessing.transformations import (
    get_all_transformations,
    TRANSFORM_REGISTRY,
    register_transform,
)
from scintilla.preprocessing.normality import check_normality
from scintilla.preprocessing.pca import run_pca, cumulative_variance_explained
from scintilla.preprocessing.aggregation import aggregate_by_patient_celltype

__all__ = [
    "get_all_transformations",
    "TRANSFORM_REGISTRY",
    "register_transform",
    "check_normality",
    "run_pca",
    "cumulative_variance_explained",
    "aggregate_by_patient_celltype",
]
