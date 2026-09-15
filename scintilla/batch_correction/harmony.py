"""Harmony batch correction."""

from __future__ import annotations

import anndata as ad
import numpy as np

from scintilla.config import RANDOM_SEED


def harmony_correct(
    adata: ad.AnnData,
    batch_key: str,
    n_components: int = 30,
    random_state: int = RANDOM_SEED,
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
    random_state:
        Random seed for PCA and Harmony optimisation.

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
        sc.tl.pca(adata, n_comps=n_components, random_state=random_state)

    X_pca = adata.obsm["X_pca"][:, :n_components]
    ho = harmonypy.run_harmony(
        X_pca, adata.obs, batch_key, random_state=random_state,
    )

    # harmonypy < 2.0 returns Z_corr as (n_components, n_cells); 2.0 returns
    # (n_cells, n_components).  Orient by the caller's own shape instead of
    # assuming either, so obsm never receives a transposed embedding.
    Z_corr = np.asarray(ho.Z_corr)
    expected = X_pca.shape
    if Z_corr.shape == expected:
        embedding = Z_corr
    elif Z_corr.shape == expected[::-1]:
        embedding = Z_corr.T
    else:
        raise ValueError(
            "harmonypy returned a corrected embedding of unexpected shape "
            f"{Z_corr.shape}; expected {expected} or {expected[::-1]}"
        )
    adata.obsm["X_pca_harmony"] = embedding
    return adata
