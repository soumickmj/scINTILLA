"""Harmony batch correction."""

from __future__ import annotations

import anndata as ad
import numpy as np


def harmony_correct(
    adata: ad.AnnData,
    batch_key: str,
    n_components: int = 30,
) -> ad.AnnData:
    """Apply Harmony batch correction to PCA embedding.

    Parameters
    ----------
    adata:
        AnnData with X_pca in obsm (computed beforehand).
    batch_key:
        Column in obs with batch labels.
    n_components:
        Number of PCA components to use.

    Returns
    -------
    AnnData with 'X_pca_harmony' added to obsm.
    """
    try:
        import harmonypy  # noqa: PLC0415
    except ImportError as exc:
        raise ImportError(
            "harmonypy is required for Harmony correction. "
            "Install with: pip install harmonypy"
        ) from exc

    adata = adata.copy()
    if "X_pca" not in adata.obsm:
        import scanpy as sc  # noqa: PLC0415
        sc.tl.pca(adata, n_comps=n_components)

    X_pca = adata.obsm["X_pca"][:, :n_components]
    ho = harmonypy.run_harmony(X_pca, adata.obs, batch_key)
    adata.obsm["X_pca_harmony"] = ho.Z_corr.T
    return adata
