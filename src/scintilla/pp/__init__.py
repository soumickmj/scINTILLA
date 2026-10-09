"""Preprocessing: normalisation and transformation, PCA, gene selection, normality checks.

Transformations write ``adata.layers[key_added]`` (default: the transformation's name) so the
raw counts survive.  ``pp.pca`` writes ``adata.obsm["X_pca"]``; ``pp.highly_variable_genes``
flags ``adata.var["highly_variable"]``.  All of them modify ``adata`` in place and return
``None``, or return a modified copy with ``copy=True``.
"""

from scintilla.feature_selection.hvg import select_hvg as highly_variable_genes
from scintilla.feature_selection.select import select_features
from scintilla.preprocessing.aggregation import aggregate_by_patient_celltype
from scintilla.preprocessing.normality import check_normality
from scintilla.preprocessing.pca import cumulative_variance_explained
from scintilla.preprocessing.pca import run_pca as pca
from scintilla.preprocessing.transformations import (
    TRANSFORM_REGISTRY,
    arcsinh_transform,
    box_cox_transform,
    get_all_transformations,
    glm_pca_transform,
    log_alpha_transform,
    log_cpm_transform,
    log_shift_hvg_z,
    log_shift_scale_by_std,
    log_shift_size_factor,
    log_shift_size_factor_hvg,
    log_shift_size_factor_z,
    normalise_scran,
    normalise_tmm,
    pearson_residuals_transform,
    register_transform,
    sanity_transform,
)

__all__ = [
    "pca",
    "cumulative_variance_explained",
    "highly_variable_genes",
    "select_features",
    "check_normality",
    "aggregate_by_patient_celltype",
    "TRANSFORM_REGISTRY",
    "get_all_transformations",
    "register_transform",
    "log_shift_size_factor",
    "arcsinh_transform",
    "log_alpha_transform",
    "log_cpm_transform",
    "log_shift_scale_by_std",
    "log_shift_size_factor_hvg",
    "log_shift_size_factor_z",
    "log_shift_hvg_z",
    "normalise_scran",
    "normalise_tmm",
    "box_cox_transform",
    "pearson_residuals_transform",
    "glm_pca_transform",
    "sanity_transform",
]
