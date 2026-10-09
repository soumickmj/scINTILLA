"""Data transformations / normalisation for single-cell data.

Each transformation exists in two forms:

* the public function, for example :func:`log_shift_size_factor`, follows the
  scintilla API contract: ``fn(adata, *, layer=None, key_added=None, replace_x=False,
  copy=False)`` reads ``adata.X`` (or ``adata.layers[layer]``) and **writes the result
  to ``adata.layers[key_added]``**, so the raw counts survive.  Transformations that
  change the set of variables (the ``*_hvg*`` variants and ``glm_pca_transform``) cannot
  be stored as a layer and return a new AnnData instead;
* the *pure* form, ``TRANSFORM_REGISTRY[name](data) -> AnnData``, which returns a
  transformed copy with the result in ``X``.  The benchmarks use this form.

Sparse input is kept sparse for the zero-preserving transformations
(``log_shift_size_factor``, ``arcsinh_transform``, ``log_alpha_transform``,
``log_cpm_transform`` and ``normalise_scran``); the others centre or invert the data
and therefore need a dense matrix.
"""

from __future__ import annotations

import inspect
from typing import Callable, Dict, Optional, Union

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.stats import boxcox

from scintilla._compat import record_params
from scintilla.config import RANDOM_SEED
from scintilla.io.loaders import ensure_anndata

# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

TRANSFORM_REGISTRY: Dict[str, Callable] = {}


def register_transform(name: str) -> Callable:
    """Register a *pure* transformation function (``data -> AnnData``) under ``name``.

    Parameters
    ----------
    name
        Name under which the function is registered.
    """

    def decorator(fn: Callable) -> Callable:
        TRANSFORM_REGISTRY[name] = fn
        return fn

    return decorator


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _to_matrix(data: Union[pd.DataFrame, ad.AnnData], dense: bool = True):
    """Return ``(adata, X)``; ``X`` is float64, dense unless ``dense=False`` and X is sparse."""
    adata = ensure_anndata(data)
    X = adata.X
    if sparse.issparse(X):
        if dense:
            return adata, X.toarray().astype(np.float64)
        return adata, X.tocsr(copy=True).astype(np.float64)
    if hasattr(X, "toarray"):
        return adata, X.toarray().astype(np.float64)
    return adata, np.asarray(X).astype(np.float64)


def _row_of_each_entry(X: sparse.csr_matrix) -> np.ndarray:
    """Row index of every stored entry of a CSR matrix (to scale ``X.data`` per cell)."""
    return np.repeat(np.arange(X.shape[0]), np.diff(X.indptr))


def _size_factors(X, floor_zero: bool = True) -> np.ndarray:
    """Per-cell totals, with empty cells mapped to 1 so the division is defined."""
    totals = np.asarray(X.sum(axis=1)).ravel().astype(np.float64)
    return np.where(totals == 0, 1.0, totals) if floor_zero else totals


def _wrap(adata: ad.AnnData, X_new, var_names=None) -> ad.AnnData:
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
def _log_shift_size_factor(data: Union[pd.DataFrame, ad.AnnData]) -> ad.AnnData:
    """log(x / size_factor + 1) where size_factor is per-cell total count."""
    adata, X = _to_matrix(data, dense=False)
    if sparse.issparse(X):
        X.data = np.log1p(X.data / _size_factors(X)[_row_of_each_entry(X)])
        return _wrap(adata, X)
    size_factors = X.sum(axis=1, keepdims=True)
    size_factors = np.where(size_factors == 0, 1.0, size_factors)
    X_t = np.log1p(X / size_factors)
    return _wrap(adata, X_t)


@register_transform("arcsinh_transform")
def _arcsinh_transform(data: Union[pd.DataFrame, ad.AnnData], alpha: float = 0.05) -> ad.AnnData:
    """arcsinh(alpha * x) transform."""
    adata, X = _to_matrix(data, dense=False)
    if sparse.issparse(X):
        X.data = np.arcsinh(alpha * X.data)
        return _wrap(adata, X)
    X_t = np.arcsinh(alpha * X)
    return _wrap(adata, X_t)


@register_transform("log_alpha_transform")
def _log_alpha_transform(data: Union[pd.DataFrame, ad.AnnData], alpha: float = 0.05) -> ad.AnnData:
    """log(alpha * x + 1) transform."""
    adata, X = _to_matrix(data, dense=False)
    if sparse.issparse(X):
        X.data = np.log1p(alpha * X.data)
        return _wrap(adata, X)
    X_t = np.log1p(alpha * X)
    return _wrap(adata, X_t)


@register_transform("log_cpm_transform")
def _log_cpm_transform(data: Union[pd.DataFrame, ad.AnnData]) -> ad.AnnData:
    """log(CPM + 1) where CPM = counts per million."""
    adata, X = _to_matrix(data, dense=False)
    if sparse.issparse(X):
        X.data = np.log1p(X.data / _size_factors(X)[_row_of_each_entry(X)] * 1e6)
        return _wrap(adata, X)
    lib_sizes = X.sum(axis=1, keepdims=True)
    lib_sizes = np.where(lib_sizes == 0, 1.0, lib_sizes)
    cpm = X / lib_sizes * 1e6
    X_t = np.log1p(cpm)
    return _wrap(adata, X_t)


@register_transform("log_shift_scale_by_std")
def _log_shift_scale_by_std(data: Union[pd.DataFrame, ad.AnnData]) -> ad.AnnData:
    """log(x+1) then scale each gene by its standard deviation."""
    adata, X = _to_matrix(data)
    X_log = np.log1p(X)
    stds = X_log.std(axis=0)
    stds[stds == 0] = 1.0
    X_t = X_log / stds
    return _wrap(adata, X_t)


@register_transform("log_shift_size_factor_hvg")
def _log_shift_size_factor_hvg(
    data: Union[pd.DataFrame, ad.AnnData],
    top_frac: float = 0.35,
) -> ad.AnnData:
    """Apply log_shift_size_factor then select highly variable genes."""
    adata_t = _log_shift_size_factor(data)
    _, X = _to_matrix(adata_t)
    variances = X.var(axis=0)
    n_top = max(1, int(len(variances) * top_frac))
    top_idx = np.argsort(variances)[-n_top:]
    top_idx = np.sort(top_idx)
    X_hvg = X[:, top_idx]
    selected_genes = adata_t.var_names[top_idx]
    return _wrap(adata_t, X_hvg, var_names=selected_genes)


@register_transform("log_shift_size_factor_z")
def _log_shift_size_factor_z(data: Union[pd.DataFrame, ad.AnnData]) -> ad.AnnData:
    """log_shift_size_factor then z-score per gene."""
    adata_t = _log_shift_size_factor(data)
    _, X = _to_matrix(adata_t)
    means = X.mean(axis=0)
    stds = X.std(axis=0)
    stds[stds == 0] = 1.0
    X_t = (X - means) / stds
    return _wrap(adata_t, X_t)


@register_transform("log_shift_hvg_z")
def _log_shift_hvg_z(
    data: Union[pd.DataFrame, ad.AnnData],
    top_frac: float = 0.35,
) -> ad.AnnData:
    """log_shift + HVG selection + z-score."""
    adata_hvg = _log_shift_size_factor_hvg(data, top_frac=top_frac)
    _, X = _to_matrix(adata_hvg)
    means = X.mean(axis=0)
    stds = X.std(axis=0)
    stds[stds == 0] = 1.0
    X_t = (X - means) / stds
    return _wrap(adata_hvg, X_t)


@register_transform("normalise_scran")
def _normalise_scran(data: Union[pd.DataFrame, ad.AnnData]) -> ad.AnnData:
    """Scran-style normalisation using median-ratio / pooling approximation.

    Implementation: normalise each cell by its library size, then scale by
    the geometric mean of library sizes across cells (median-based deconvolution
    approximation).
    """
    adata, X = _to_matrix(data, dense=False)
    lib_sizes = _size_factors(X)
    # geometric mean of library sizes
    geo_mean = np.exp(np.mean(np.log(lib_sizes + 1e-8)))
    size_factors = lib_sizes / geo_mean
    if sparse.issparse(X):
        X.data = np.log1p(X.data / size_factors[_row_of_each_entry(X)])
        return _wrap(adata, X)
    X_norm = X / size_factors[:, np.newaxis]
    X_t = np.log1p(X_norm)
    return _wrap(adata, X_t)


@register_transform("normalise_tmm")
def _normalise_tmm(data: Union[pd.DataFrame, ad.AnnData]) -> ad.AnnData:
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
def _box_cox_transform(data: Union[pd.DataFrame, ad.AnnData]) -> ad.AnnData:
    """Box-Cox transform per gene. Constant or failing columns are kept unchanged.

    Uses parallel execution when joblib is available for significant
    speedups on datasets with many genes.
    """
    adata, X = _to_matrix(data)
    X_t = X.copy()

    gene_args = [(X[:, j].copy(), j) for j in range(X.shape[1])]

    try:
        from joblib import Parallel, delayed
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
def _pearson_residuals_transform(
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
        import scanpy as sc
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
def _glm_pca_transform(
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
    from sklearn.decomposition import TruncatedSVD

    adata, X = _to_matrix(data)
    lib_sizes = X.sum(axis=1, keepdims=True)
    lib_sizes = np.where(lib_sizes == 0, 1.0, lib_sizes)
    X_norm = np.log1p(X / lib_sizes * 1e4)

    n_c = min(n_components, X_norm.shape[0] - 1, X_norm.shape[1] - 1)
    svd = TruncatedSVD(n_components=n_c, random_state=random_state)
    Z = svd.fit_transform(X_norm)

    import anndata as ad
    out = ad.AnnData(X=Z.astype(np.float32))
    out.obs = adata.obs.copy()
    return out


@register_transform("sanity_transform")
def _sanity_transform(data: Union[pd.DataFrame, ad.AnnData]) -> ad.AnnData:
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


# ---------------------------------------------------------------------------
# Public API: layer-writing wrappers around the pure transformations
# ---------------------------------------------------------------------------

_PUBLIC_PARAMS_DOC = """
Parameters
----------
adata
    Annotated data matrix (raw counts expected).
layer
    Layer to read; ``None`` reads ``adata.X``.
key_added
    Name of the layer that receives the result.  Defaults to ``"{name}"``.
replace_x
    Write the result to ``adata.X`` instead of a layer (the raw values are lost).
copy
    Return a modified copy and leave ``adata`` untouched, instead of modifying
    ``adata`` in place and returning ``None``.
{extra}
Returns
-------
anndata.AnnData or None
    ``None`` when working in place; the modified copy when ``copy=True``.
    {changes_vars_note}
Notes
-----
The parameters used are recorded in ``adata.uns["scintilla"]["{name}"]``.
    """


def _make_public(name: str, pure: Callable, *, changes_vars: bool = False) -> Callable:
    """Build the public, layer-writing version of a pure transformation."""
    pure_sig = inspect.signature(pure)
    pure_params = list(pure_sig.parameters.values())
    data_name, extra_params = pure_params[0].name, pure_params[1:]
    summary = (inspect.getdoc(pure) or "").split("\n\n", 1)[0]

    def wrapper(adata, *args, layer=None, key_added=None, replace_x=False, copy=False, **kwargs):
        adata = ensure_anndata(adata)
        if layer is not None and layer not in adata.layers:
            raise KeyError(f"Layer {layer!r} not found in adata.layers.")
        source = adata
        if layer is not None:
            source = ad.AnnData(X=adata.layers[layer], obs=adata.obs, var=adata.var)
        result = pure(source, *args, **kwargs)
        bound = pure_sig.bind(source, *args, **kwargs)
        bound.apply_defaults()
        params = {k: v for k, v in bound.arguments.items() if k != data_name}

        if changes_vars:
            record_params(result, name, layer=layer, **params)
            return result

        target = adata.copy() if copy else adata
        if replace_x:
            target.X = result.X
        else:
            target.layers[key_added or name] = result.X
        record_params(target, name if replace_x else (key_added or name), layer=layer, replace_x=replace_x, **params)
        return target if copy else None

    extra_doc = "".join(
        f"{p.name}\n    Transformation parameter, default ``{p.default!r}``.\n"
        for p in extra_params
        if p.default is not inspect.Parameter.empty
    )
    note = (
        "This transformation changes the set of variables, so it cannot be stored as a layer: "
        "a new AnnData is always returned and ``adata`` is not modified."
        if changes_vars
        else ""
    )
    wrapper.__name__ = wrapper.__qualname__ = name
    wrapper.__module__ = pure.__module__
    wrapper.__doc__ = summary + "\n" + _PUBLIC_PARAMS_DOC.format(name=name, extra=extra_doc, changes_vars_note=note)
    keyword_only = inspect.Parameter.KEYWORD_ONLY
    wrapper.__signature__ = inspect.Signature(
        [
            inspect.Parameter("adata", inspect.Parameter.POSITIONAL_OR_KEYWORD),
            *extra_params,
            inspect.Parameter("layer", keyword_only, default=None, annotation=Optional[str]),
            inspect.Parameter("key_added", keyword_only, default=None, annotation=Optional[str]),
            inspect.Parameter("replace_x", keyword_only, default=False, annotation=bool),
            inspect.Parameter("copy", keyword_only, default=False, annotation=bool),
        ],
        return_annotation=Optional[ad.AnnData],
    )
    return wrapper


log_shift_size_factor = _make_public("log_shift_size_factor", _log_shift_size_factor)
arcsinh_transform = _make_public("arcsinh_transform", _arcsinh_transform)
log_alpha_transform = _make_public("log_alpha_transform", _log_alpha_transform)
log_cpm_transform = _make_public("log_cpm_transform", _log_cpm_transform)
log_shift_scale_by_std = _make_public("log_shift_scale_by_std", _log_shift_scale_by_std)
log_shift_size_factor_hvg = _make_public("log_shift_size_factor_hvg", _log_shift_size_factor_hvg, changes_vars=True)
log_shift_size_factor_z = _make_public("log_shift_size_factor_z", _log_shift_size_factor_z)
log_shift_hvg_z = _make_public("log_shift_hvg_z", _log_shift_hvg_z, changes_vars=True)
normalise_scran = _make_public("normalise_scran", _normalise_scran)
normalise_tmm = _make_public("normalise_tmm", _normalise_tmm)
box_cox_transform = _make_public("box_cox_transform", _box_cox_transform)
pearson_residuals_transform = _make_public("pearson_residuals_transform", _pearson_residuals_transform)
glm_pca_transform = _make_public("glm_pca_transform", _glm_pca_transform, changes_vars=True)
sanity_transform = _make_public("sanity_transform", _sanity_transform)
