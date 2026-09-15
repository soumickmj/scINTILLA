"""PCA utilities using scanpy."""

from __future__ import annotations

from typing import Optional, Tuple, Union

import anndata as ad
import numpy as np
import pandas as pd

from scintilla.config import DEFAULT_N_PCA_COMPS, DEFAULT_VARIANCE_THRESHOLD, RANDOM_SEED
from scintilla.io.loaders import ensure_anndata


def run_pca(
    data: Union[pd.DataFrame, ad.AnnData],
    n_comps: int = DEFAULT_N_PCA_COMPS,
    variance_threshold: Optional[float] = None,
    auto_components: Optional[str] = None,
    mp_sigma_method: str = "median",
    random_state: int = RANDOM_SEED,
) -> ad.AnnData:
    """Run PCA using scanpy and store results in adata.obsm['X_pca'].

    If *variance_threshold* is given, automatically select the number of
    components needed to explain that fraction of variance.

    Parameters
    ----------
    auto_components:
        Data-adaptive component selection method.  Overrides *n_comps*
        and *variance_threshold* when set.

        - ``"gavish_donoho"`` — Gavish-Donoho optimal hard threshold for
          singular values under a noise model.
        - ``"marchenko_pastur"`` — retain eigenvalues exceeding the
          Marchenko-Pastur bulk edge.
        - ``None`` (default) — use *n_comps* / *variance_threshold* as
          before.
    mp_sigma_method:
        Noise-variance estimation method passed to
        ``marchenko_pastur_cutoff`` when ``auto_components="marchenko_pastur"``.
        ``"median"`` (default) or ``"trimmed_mean"``.
    """
    import scanpy as sc  # noqa: PLC0415

    adata = ensure_anndata(data)
    max_comps = min(adata.n_obs - 1, adata.n_vars - 1)
    if max_comps < 1:
        raise ValueError(
            f"PCA requires at least 2 observations and 2 variables, "
            f"but got n_obs={adata.n_obs}, n_vars={adata.n_vars}."
        )

    if auto_components is not None:
        from scintilla.statistical_tests.adaptive import (  # noqa: PLC0415
            gavish_donoho_threshold,
            marchenko_pastur_cutoff,
        )
        # Run PCA with maximum feasible components first
        max_fit = min(max_comps, 100)
        sc.pp.pca(adata, n_comps=max_fit, random_state=random_state)
        svd_solver_key = adata.uns.get("pca", {})
        # Retrieve singular values; scanpy stores variance, convert back
        var_ratio = adata.uns["pca"]["variance_ratio"]
        variance = adata.uns["pca"]["variance"]
        singular_values = np.sqrt(variance * (adata.n_obs - 1))

        n_obs, n_vars = adata.n_obs, adata.n_vars
        if auto_components == "gavish_donoho":
            n_keep = gavish_donoho_threshold(singular_values, n_obs, n_vars)
        elif auto_components == "marchenko_pastur":
            n_keep = marchenko_pastur_cutoff(
                singular_values, n_obs, n_vars,
                sigma_method=mp_sigma_method,
            )
        else:
            raise ValueError(
                f"Unknown auto_components method: {auto_components!r}. "
                "Supported: 'gavish_donoho', 'marchenko_pastur'."
            )
        n_keep = max(2, min(n_keep, max_fit))
        adata.obsm["X_pca"] = adata.obsm["X_pca"][:, :n_keep]
        adata.uns["pca"]["auto_n_comps"] = n_keep
        adata.uns["pca"]["auto_method"] = auto_components
        return adata

    n_comps = min(n_comps, max_comps)
    sc.pp.pca(adata, n_comps=n_comps, random_state=random_state)

    if variance_threshold is not None:
        cum_var = cumulative_variance_explained(adata)
        # find minimum n_comps to exceed threshold
        n_needed = int(np.searchsorted(cum_var, variance_threshold)) + 1
        n_needed = max(2, min(n_needed, n_comps))
        adata.obsm["X_pca"] = adata.obsm["X_pca"][:, :n_needed]

    return adata


def cumulative_variance_explained(adata: ad.AnnData) -> np.ndarray:
    """Return cumulative variance explained by PCs.

    Requires that sc.pp.pca has already been run (stores variance_ratio in uns).
    """
    if "pca" not in adata.uns or "variance_ratio" not in adata.uns["pca"]:
        raise ValueError("PCA has not been run. Call run_pca first.")
    return np.cumsum(adata.uns["pca"]["variance_ratio"])
