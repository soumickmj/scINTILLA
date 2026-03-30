"""Scanorama batch correction."""

from __future__ import annotations

import anndata as ad
import numpy as np


def scanorama_correct(
    adata: ad.AnnData,
    batch_key: str,
) -> ad.AnnData:
    """Apply Scanorama batch correction.

    Parameters
    ----------
    adata:
        AnnData object.
    batch_key:
        Column in obs with batch labels.

    Returns
    -------
    AnnData with 'X_scanorama' in obsm.
    """
    try:
        import scanorama  # noqa: PLC0415
    except ImportError as exc:
        raise ImportError(
            "scanorama is required. Install with: pip install scanorama"
        ) from exc

    adata = adata.copy()
    batches = adata.obs[batch_key].unique().tolist()
    adatas = [adata[adata.obs[batch_key] == b].copy() for b in batches]

    corrected, _ = scanorama.correct_scanpy(adatas, return_dimred=True)
    # Reconstruct in original order
    import pandas as pd  # noqa: PLC0415
    order = []
    for b, adata_b in zip(batches, adatas):
        order.extend(adata_b.obs_names.tolist())

    emb = np.vstack([c.obsm["X_scanorama"] for c in corrected])
    # Reorder to match adata.obs_names
    idx_map = {name: i for i, name in enumerate(order)}
    reorder = [idx_map[name] for name in adata.obs_names]
    adata.obsm["X_scanorama"] = emb[reorder]
    return adata
