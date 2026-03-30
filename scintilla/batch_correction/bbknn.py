"""BBKNN batch correction."""

from __future__ import annotations

import anndata as ad


def bbknn_correct(
    adata: ad.AnnData,
    batch_key: str,
    n_pcs: int = 30,
) -> ad.AnnData:
    """Apply BBKNN (Batch Balanced kNN) batch correction.

    Parameters
    ----------
    adata:
        AnnData with X_pca in obsm.
    batch_key:
        Column in obs with batch labels.
    n_pcs:
        Number of PCs to use.

    Returns
    -------
    AnnData with BBKNN connectivities.
    """
    try:
        import bbknn  # noqa: PLC0415
    except ImportError as exc:
        raise ImportError(
            "bbknn is required. Install with: pip install bbknn"
        ) from exc

    import scanpy as sc  # noqa: PLC0415

    adata = adata.copy()
    if "X_pca" not in adata.obsm:
        sc.tl.pca(adata, n_comps=n_pcs)

    bbknn.bbknn(adata, batch_key=batch_key, n_pcs=n_pcs)
    return adata
