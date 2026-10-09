"""Preprocessing sub-package."""

from scintilla.preprocessing.aggregation import aggregate_by_patient_celltype
from scintilla.preprocessing.normality import check_normality
from scintilla.preprocessing.pca import cumulative_variance_explained, run_pca
from scintilla.preprocessing.transformations import (
    TRANSFORM_REGISTRY,
    get_all_transformations,
    register_transform,
)

__all__ = [
    "get_all_transformations",
    "TRANSFORM_REGISTRY",
    "register_transform",
    "check_normality",
    "run_pca",
    "cumulative_variance_explained",
    "aggregate_by_patient_celltype",
]
