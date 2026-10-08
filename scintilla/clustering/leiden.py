"""Leiden community-detection clustering via scanpy."""

from __future__ import annotations

from typing import Union

import anndata as ad
import numpy as np
import pandas as pd

from scintilla.config import RANDOM_SEED
from scintilla.io.loaders import ensure_anndata


def leiden_clustering(
    data: Union[pd.DataFrame, ad.AnnData],
    resolution: float = 1.0,
    use_rep: str = "X_pca",
    random_state: int = RANDOM_SEED,
) -> np.ndarray:
    """Run Leiden clustering via scanpy.

    Parameters
    ----------
    data:
        Input data (AnnData preferred; must have *use_rep* in obsm if AnnData).
    resolution:
        Leiden resolution parameter.
    use_rep:
        Key in obsm to use as representation for neighbour graph.

    Returns
    -------
    np.ndarray of cluster labels (int).
    """
    try:
        import scanpy as sc  # noqa: PLC0415
        import leidenalg  # noqa: PLC0415, F401
    except ImportError as exc:
        raise ImportError(
            "leidenalg and scanpy are required for Leiden clustering. "
            "Install with: pip install leidenalg python-igraph scanpy"
        ) from exc

    adata = ensure_anndata(data)

    if use_rep not in adata.obsm:
        # Run PCA if representation not available
        import scintilla.preprocessing.pca as pca_mod  # noqa: PLC0415
        adata = pca_mod.run_pca(adata, random_state=random_state)
        use_rep = "X_pca"

    sc.pp.neighbors(adata, use_rep=use_rep, random_state=random_state)
    sc.tl.leiden(adata, resolution=resolution, random_state=random_state, flavor="igraph")
    labels = adata.obs["leiden"].astype(int).values
    return labels
