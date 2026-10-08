"""ComBat batch correction via scanpy."""

from __future__ import annotations

import anndata as ad


def combat_correct(
    adata: ad.AnnData,
    batch_key: str,
) -> ad.AnnData:
    """Apply ComBat batch correction via scanpy.pp.combat.

    Parameters
    ----------
    adata:
        AnnData object.
    batch_key:
        Column in obs with batch labels.

    Returns
    -------
    Batch-corrected AnnData.
    """
    try:
        import scanpy as sc  # noqa: PLC0415
    except ImportError as exc:
        raise ImportError("scanpy is required. Install with: pip install scanpy") from exc

    adata = adata.copy()
    sc.pp.combat(adata, key=batch_key)
    return adata
