"""Data transformations / normalisation for single-cell data.

All transforms accept either a pd.DataFrame or an AnnData and return an AnnData
with the transformed expression matrix stored in X.
"""

from __future__ import annotations

from typing import Callable, Dict, Optional, Union

import anndata as ad
import numpy as np
import pandas as pd
from scipy.stats import boxcox

from scintilla.io.loaders import ensure_anndata
from scintilla.config import RANDOM_SEED

# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

TRANSFORM_REGISTRY: Dict[str, Callable] = {}


def register_transform(name: str) -> Callable:
    """Decorator to register a transformation function."""

    def decorator(fn: Callable) -> Callable:
        TRANSFORM_REGISTRY[name] = fn
        return fn

    return decorator


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _to_matrix(data: Union[pd.DataFrame, ad.AnnData]) -> tuple[ad.AnnData, np.ndarray]:
    adata = ensure_anndata(data)
    X = adata.X if not hasattr(adata.X, "toarray") else adata.X.toarray()
    return adata, X.astype(np.float64)


def _wrap(adata: ad.AnnData, X_new: np.ndarray, var_names=None) -> ad.AnnData:
    """Create a new AnnData with transformed matrix, preserving all metadata.

    Copies obs, var, obsm, varm, obsp, uns, and layers from *adata*.  When
    *var_names* is given the output is first subsetted to those variables so
    that var-aligned fields (var, varm, layers) have the correct shape.
    """
    if var_names is not None:
        out = adata[:, list(var_names)].copy()
    else:
        out = adata.copy()
    out.X = X_new.astype(np.float32)
    return out


# ---------------------------------------------------------------------------
# Transformations
# ---------------------------------------------------------------------------

@register_transform("log_shift_size_factor")
def log_shift_size_factor(data: Union[pd.DataFrame, ad.AnnData]) -> ad.AnnData:
    """log(x / size_factor + 1) where size_factor is per-cell total count."""
    adata, X = _to_matrix(data)
    size_factors = X.sum(axis=1, keepdims=True)
    size_factors = np.where(size_factors == 0, 1.0, size_factors)
    X_t = np.log1p(X / size_factors)
    return _wrap(adata, X_t)


@register_transform("arcsinh_transform")
def arcsinh_transform(data: Union[pd.DataFrame, ad.AnnData], alpha: float = 0.05) -> ad.AnnData:
    """arcsinh(alpha * x) transform."""
    adata, X = _to_matrix(data)
    X_t = np.arcsinh(alpha * X)
    return _wrap(adata, X_t)


@register_transform("log_alpha_transform")
def log_alpha_transform(data: Union[pd.DataFrame, ad.AnnData], alpha: float = 0.05) -> ad.AnnData:
    """log(alpha * x + 1) transform."""
    adata, X = _to_matrix(data)
    X_t = np.log1p(alpha * X)
    return _wrap(adata, X_t)


@register_transform("log_cpm_transform")
def log_cpm_transform(data: Union[pd.DataFrame, ad.AnnData]) -> ad.AnnData:
    """log(CPM + 1) where CPM = counts per million."""
    adata, X = _to_matrix(data)
    lib_sizes = X.sum(axis=1, keepdims=True)
    lib_sizes = np.where(lib_sizes == 0, 1.0, lib_sizes)
    cpm = X / lib_sizes * 1e6
    X_t = np.log1p(cpm)
    return _wrap(adata, X_t)


@register_transform("log_shift_scale_by_std")
def log_shift_scale_by_std(data: Union[pd.DataFrame, ad.AnnData]) -> ad.AnnData:
    """log(x+1) then scale each gene by its standard deviation."""
    adata, X = _to_matrix(data)
    X_log = np.log1p(X)
    stds = X_log.std(axis=0)
    stds[stds == 0] = 1.0
    X_t = X_log / stds
    return _wrap(adata, X_t)


@register_transform("log_shift_size_factor_hvg")
def log_shift_size_factor_hvg(
    data: Union[pd.DataFrame, ad.AnnData],
    top_frac: float = 0.35,
) -> ad.AnnData:
    """Apply log_shift_size_factor then select highly variable genes."""
    adata_t = log_shift_size_factor(data)
    _, X = _to_matrix(adata_t)
    variances = X.var(axis=0)
    n_top = max(1, int(len(variances) * top_frac))
    top_idx = np.argsort(variances)[-n_top:]
    top_idx = np.sort(top_idx)
    X_hvg = X[:, top_idx]
    selected_genes = adata_t.var_names[top_idx]
    return _wrap(adata_t, X_hvg, var_names=selected_genes)


@register_transform("log_shift_size_factor_z")
def log_shift_size_factor_z(data: Union[pd.DataFrame, ad.AnnData]) -> ad.AnnData:
    """log_shift_size_factor then z-score per gene."""
    adata_t = log_shift_size_factor(data)
    _, X = _to_matrix(adata_t)
    means = X.mean(axis=0)
    stds = X.std(axis=0)
    stds[stds == 0] = 1.0
    X_t = (X - means) / stds
    return _wrap(adata_t, X_t)


@register_transform("log_shift_hvg_z")
def log_shift_hvg_z(
    data: Union[pd.DataFrame, ad.AnnData],
    top_frac: float = 0.35,
) -> ad.AnnData:
    """log_shift + HVG selection + z-score."""
    adata_hvg = log_shift_size_factor_hvg(data, top_frac=top_frac)
    _, X = _to_matrix(adata_hvg)
    means = X.mean(axis=0)
    stds = X.std(axis=0)
    stds[stds == 0] = 1.0
    X_t = (X - means) / stds
    return _wrap(adata_hvg, X_t)


@register_transform("normalise_scran")
def normalise_scran(data: Union[pd.DataFrame, ad.AnnData]) -> ad.AnnData:
    """Scran-style normalisation using median-ratio / pooling approximation.

    Implementation: normalise each cell by its library size, then scale by
    the geometric mean of library sizes across cells (median-based deconvolution
    approximation).
    """
    adata, X = _to_matrix(data)
    lib_sizes = X.sum(axis=1)
    lib_sizes = np.where(lib_sizes == 0, 1.0, lib_sizes)
    # geometric mean of library sizes
    geo_mean = np.exp(np.mean(np.log(lib_sizes + 1e-8)))
    size_factors = lib_sizes / geo_mean
    X_norm = X / size_factors[:, np.newaxis]
    X_t = np.log1p(X_norm)
    return _wrap(adata, X_t)


@register_transform("normalise_tmm")
def normalise_tmm(data: Union[pd.DataFrame, ad.AnnData]) -> ad.AnnData:
    """TMM normalisation (Robinson & Oshlack 2010).

    Compute M-values and A-values relative to a reference sample,
    trim 30% from both tails of M and A, compute weighted mean of
    trimmed M-values as normalisation factor.
    """
    adata, X = _to_matrix(data)
    # add pseudocount
    pseudo = 0.5
    Y = X + pseudo

    lib_sizes = Y.sum(axis=1)  # (n_cells,)
    # Use cell with 75th-percentile library size as reference
    ref_idx = int(np.argsort(lib_sizes)[int(0.75 * len(lib_sizes))])
    ref = Y[ref_idx, :]
    ref_lib = lib_sizes[ref_idx]

    tmm_factors = np.ones(Y.shape[0])
    for i in range(Y.shape[0]):
        if i == ref_idx:
            continue
        obs = Y[i, :]
        obs_lib = lib_sizes[i]

        # keep genes expressed in both
        mask = (obs > 0) & (ref > 0)
        if mask.sum() < 5:
            continue

        obs_m = obs[mask]
        ref_m = ref[mask]

        # M = log2(obs/ref) adjusted for library size
        M = np.log2(obs_m / obs_lib) - np.log2(ref_m / ref_lib)
        # A = 0.5 * (log2(obs) + log2(ref))
        A = 0.5 * (np.log2(obs_m / obs_lib) + np.log2(ref_m / ref_lib))

        # Trim 30% from M and A tails
        n = len(M)
        lo, hi = int(0.30 * n), int(0.70 * n)
        m_order = np.argsort(M)
        a_order = np.argsort(A)
        keep = np.intersect1d(m_order[lo:hi], a_order[lo:hi])
        if len(keep) < 3:
            keep = m_order[lo:hi]

        # weights = 1/var(log-ratio)
        w_obs = (obs_lib - obs_m[keep]) / (obs_lib * obs_m[keep])
        w_ref = (ref_lib - ref_m[keep]) / (ref_lib * ref_m[keep])
        weights = 1.0 / (w_obs + w_ref + 1e-8)

        tmm = np.sum(weights * M[keep]) / np.sum(weights)
        tmm_factors[i] = 2 ** tmm

    # Normalise to geometric mean of factors = 1
    geo_mean_f = np.exp(np.mean(np.log(tmm_factors + 1e-8)))
    tmm_factors /= geo_mean_f

    X_norm = X / tmm_factors[:, np.newaxis]
    X_t = np.log1p(X_norm)
    return _wrap(adata, X_t)


def _boxcox_single_gene(args):
    """Box-Cox transform a single gene column. Used by box_cox_transform."""
    col, j = args
    shift = 0.0
    if col.min() <= 0:
        shift = -col.min() + 1.0
    col_pos = col + shift
    if col_pos.std() < 1e-10:
        return j, col  # constant column – keep unchanged
    try:
        col_t, _ = boxcox(col_pos)
        return j, col_t
    except Exception:
        return j, col  # keep original if Box-Cox fails


@register_transform("box_cox_transform")
def box_cox_transform(data: Union[pd.DataFrame, ad.AnnData]) -> ad.AnnData:
    """Box-Cox transform per gene. Constant or failing columns are kept unchanged.

    Uses parallel execution when joblib is available for significant
    speedups on datasets with many genes.
    """
    adata, X = _to_matrix(data)
    X_t = X.copy()

    gene_args = [(X[:, j].copy(), j) for j in range(X.shape[1])]

    try:
        from joblib import Parallel, delayed  # noqa: PLC0415
        results = Parallel(n_jobs=-1, prefer="threads")(
            delayed(_boxcox_single_gene)(a) for a in gene_args
        )
    except ImportError:
        results = [_boxcox_single_gene(a) for a in gene_args]

    for j, col_t in results:
        X_t[:, j] = col_t

    return _wrap(adata, X_t)


def get_all_transformations() -> Dict[str, Callable]:
    """Return a copy of the full transform registry."""
    return dict(TRANSFORM_REGISTRY)


@register_transform("pearson_residuals_transform")
def pearson_residuals_transform(
    data: Union[pd.DataFrame, ad.AnnData], theta: float = 100.0
) -> ad.AnnData:
    """Analytic Pearson residuals transform.

    Uses scanpy.experimental.pp.normalize_pearson_residuals if available,
    otherwise computes analytic Pearson residuals manually.

    Parameters
    ----------
    data:
        Input data (raw counts).
    theta:
        NB dispersion parameter.

    Returns
    -------
    AnnData with Pearson residuals in X.
    """
    adata, X = _to_matrix(data)
    try:
        import scanpy as sc  # noqa: PLC0415
        adata_copy = adata.copy()
        adata_copy.X = X.astype(np.float32)
        sc.experimental.pp.normalize_pearson_residuals(adata_copy, theta=theta)
        return adata_copy
    except Exception:
        # Manual analytic Pearson residuals
        n_cells, n_genes = X.shape
        cell_counts = X.sum(axis=1, keepdims=True)
        gene_counts = X.sum(axis=0, keepdims=True)
        total = X.sum()
        mu = cell_counts * gene_counts / (total + 1e-10)
        var_nb = mu + mu ** 2 / theta
        residuals = (X - mu) / np.sqrt(var_nb + 1e-10)
        # clip to sqrt(n_cells)
        clip_val = np.sqrt(n_cells)
        residuals = np.clip(residuals, -clip_val, clip_val)
        return _wrap(adata, residuals)


@register_transform("glm_pca_transform")
def glm_pca_transform(
    data: Union[pd.DataFrame, ad.AnnData], n_components: int = 50,
    random_state: int = RANDOM_SEED,
) -> ad.AnnData:
    """Poisson GLM-PCA approximation via TruncatedSVD on log1p-normalised data.

    True GLM-PCA requires a specialised library; this is a practical
    approximation using TruncatedSVD on log-normalised counts.

    Parameters
    ----------
    data:
        Input data (raw counts).
    n_components:
        Number of latent dimensions.

    Returns
    -------
    AnnData with GLM-PCA approximation in X (n_cells x n_components).
    """
    from sklearn.decomposition import TruncatedSVD  # noqa: PLC0415

    adata, X = _to_matrix(data)
    lib_sizes = X.sum(axis=1, keepdims=True)
    lib_sizes = np.where(lib_sizes == 0, 1.0, lib_sizes)
    X_norm = np.log1p(X / lib_sizes * 1e4)

    n_c = min(n_components, X_norm.shape[0] - 1, X_norm.shape[1] - 1)
    svd = TruncatedSVD(n_components=n_c, random_state=random_state)
    Z = svd.fit_transform(X_norm)

    import anndata as ad  # noqa: PLC0415
    out = ad.AnnData(X=Z.astype(np.float32))
    out.obs = adata.obs.copy()
    return out


@register_transform("sanity_transform")
def sanity_transform(data: Union[pd.DataFrame, ad.AnnData]) -> ad.AnnData:
    """Bayesian estimation approximation (SANITY-like).

    Normalises by library size then log-transforms with a Bayesian
    pseudocount derived from n_cells / total_count.

    Parameters
    ----------
    data:
        Input data.

    Returns
    -------
    AnnData with SANITY-approximated expression.
    """
    adata, X = _to_matrix(data)
    n_cells = X.shape[0]
    total_count = X.sum()
    pseudocount = n_cells / (total_count + 1e-10)
    lib_sizes = X.sum(axis=1, keepdims=True)
    lib_sizes = np.where(lib_sizes == 0, 1.0, lib_sizes)
    X_t = np.log(X / lib_sizes + pseudocount)
    return _wrap(adata, X_t)
