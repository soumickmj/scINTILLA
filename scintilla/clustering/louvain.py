"""Louvain community-detection clustering via scanpy."""

from __future__ import annotations

from typing import Union

import anndata as ad
import numpy as np
import pandas as pd

from scintilla.config import RANDOM_SEED
from scintilla.io.loaders import ensure_anndata


def louvain_clustering(
    data: Union[pd.DataFrame, ad.AnnData],
    resolution: float = 1.0,
    use_rep: str = "X_pca",
) -> np.ndarray:
    """Run Louvain clustering via scanpy.

    Parameters
    ----------
    data:
        Input data (AnnData preferred; must have *use_rep* in obsm if AnnData).
    resolution:
        Louvain resolution parameter.
    use_rep:
        Key in obsm to use as representation for neighbour graph.

    Returns
    -------
    np.ndarray of cluster labels (int).
    """
    try:
        import scanpy as sc  # noqa: PLC0415
    except ImportError as exc:
        raise ImportError(
            "scanpy is required for Louvain clustering. "
            "Install with: pip install scanpy"
        ) from exc

    try:
        import louvain  # noqa: PLC0415, F401
    except ImportError:
        pass  # scanpy will raise a more informative error if needed

    adata = ensure_anndata(data)

    if use_rep not in adata.obsm:
        import scintilla.preprocessing.pca as pca_mod  # noqa: PLC0415
        adata = pca_mod.run_pca(adata)
        use_rep = "X_pca"

    sc.pp.neighbors(adata, use_rep=use_rep, random_state=RANDOM_SEED)
    sc.tl.louvain(adata, resolution=resolution, random_state=RANDOM_SEED)
    labels = adata.obs["louvain"].astype(int).values
    return labels
