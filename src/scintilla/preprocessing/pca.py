"""PCA utilities using scanpy."""

from __future__ import annotations

from typing import Optional, Union

import anndata as ad
import numpy as np
import pandas as pd

from scintilla._compat import finish, prepare, record_params
from scintilla.config import DEFAULT_N_PCA_COMPS, RANDOM_SEED


def run_pca(
    adata: Union[ad.AnnData, pd.DataFrame],
    n_comps: int = DEFAULT_N_PCA_COMPS,
    variance_threshold: Optional[float] = None,
    auto_components: Optional[str] = None,
    mp_sigma_method: str = "median",
    random_state: int = RANDOM_SEED,
    *,
    layer: Optional[str] = None,
    key_added: str = "X_pca",
    copy: bool = False,
) -> Optional[ad.AnnData]:
    """Run PCA using scanpy and store the embedding in ``adata.obsm[key_added]``.

    If *variance_threshold* is given, automatically select the number of
    components needed to explain that fraction of variance.  Sparse input is
    passed to scanpy unchanged.

    Parameters
    ----------
    adata
        Annotated data matrix.  A DataFrame is converted to AnnData, in which case
        the converted object is returned.
    n_comps
        Number of principal components (capped at ``min(n_obs, n_vars) - 1``).
    variance_threshold
        Keep the smallest number of components whose cumulative explained variance
        reaches this fraction.
    auto_components
        Data-adaptive component selection method.  Overrides *n_comps*
        and *variance_threshold* when set.

        - ``"gavish_donoho"``: Gavish-Donoho optimal hard threshold for
          singular values under a noise model.
        - ``"marchenko_pastur"``: retain eigenvalues exceeding the
          Marchenko-Pastur bulk edge.
        - ``None`` (default): use *n_comps* / *variance_threshold*.
    mp_sigma_method
        Noise-variance estimation method passed to
        :func:`~scintilla.statistical_tests.adaptive.marchenko_pastur_cutoff` when
        ``auto_components="marchenko_pastur"``: ``"median"`` (default) or
        ``"trimmed_mean"``.
    random_state
        Seed for the randomised solver.
    layer
        Layer to decompose; ``None`` uses ``.X``.
    key_added
        Key of the embedding in ``adata.obsm``.  The default ``"X_pca"`` is what the
        downstream scintilla functions look for.  A different key leaves any
        existing ``X_pca``, ``PCs`` and ``uns["pca"]`` untouched and stores the
        loadings in ``varm[key_added + "_loadings"]`` and the variance information in
        ``uns[key_added]``.
    copy
        Return a modified copy instead of modifying ``adata`` in place.

    Returns
    -------
    anndata.AnnData or None
        ``None`` when working in place on an AnnData, the modified AnnData otherwise.
        The component actually kept is recorded in ``adata.uns["pca"]`` and
        ``adata.uns["scintilla"]["pca"]``.
    """
    import scanpy as sc

    adata, give_back = prepare(adata, copy=copy)
    max_comps = min(adata.n_obs - 1, adata.n_vars - 1)
    if max_comps < 1:
        raise ValueError(
            f"PCA requires at least 2 observations and 2 variables, "
            f"but got n_obs={adata.n_obs}, n_vars={adata.n_vars}."
        )

    saved = {}
    if key_added != "X_pca":
        saved = {
            "obsm": adata.obsm["X_pca"] if "X_pca" in adata.obsm else None,
            "varm": adata.varm["PCs"] if "PCs" in adata.varm else None,
            "uns": adata.uns["pca"] if "pca" in adata.uns else None,
        }

    def _scanpy_pca(n):
        sc.pp.pca(adata, n_comps=n, layer=layer, random_state=random_state)

    if auto_components is not None:
        from scintilla.statistical_tests.adaptive import (
            gavish_donoho_threshold,
            marchenko_pastur_cutoff,
        )
        # Run PCA with maximum feasible components first
        max_fit = min(max_comps, 100)
        _scanpy_pca(max_fit)
        # scanpy stores variance; convert back to singular values
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
        n_final = n_keep
    else:
        n_comps = min(n_comps, max_comps)
        _scanpy_pca(n_comps)
        n_final = n_comps

        if variance_threshold is not None:
            cum_var = cumulative_variance_explained(adata)
            # find minimum n_comps to exceed threshold
            n_needed = int(np.searchsorted(cum_var, variance_threshold)) + 1
            n_needed = max(2, min(n_needed, n_comps))
            adata.obsm["X_pca"] = adata.obsm["X_pca"][:, :n_needed]
            n_final = n_needed

    if key_added != "X_pca":
        adata.obsm[key_added] = adata.obsm.pop("X_pca")
        adata.varm[key_added + "_loadings"] = adata.varm.pop("PCs")
        adata.uns[key_added] = adata.uns.pop("pca")
        for slot, value in (("obsm", saved["obsm"]), ("varm", saved["varm"]), ("uns", saved["uns"])):
            if value is None:
                continue
            {"obsm": adata.obsm, "varm": adata.varm, "uns": adata.uns}[slot][
                {"obsm": "X_pca", "varm": "PCs", "uns": "pca"}[slot]
            ] = value

    record_params(
        adata, "pca", n_comps=n_comps, n_comps_kept=n_final, variance_threshold=variance_threshold,
        auto_components=auto_components, mp_sigma_method=mp_sigma_method, random_state=random_state,
        layer=layer, key_added=key_added,
    )
    return finish(adata, give_back)


def cumulative_variance_explained(adata: ad.AnnData) -> np.ndarray:
    """Return cumulative variance explained by PCs.

    Requires that sc.pp.pca has already been run (stores variance_ratio in uns).

    Parameters
    ----------
    adata
        Annotated data matrix.
    """
    if "pca" not in adata.uns or "variance_ratio" not in adata.uns["pca"]:
        raise ValueError("PCA has not been run. Call run_pca first.")
    return np.cumsum(adata.uns["pca"]["variance_ratio"])
