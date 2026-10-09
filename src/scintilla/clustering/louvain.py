"""Louvain community-detection clustering via scanpy."""

from __future__ import annotations

from typing import Optional, Union

import anndata as ad
import numpy as np
import pandas as pd

from scintilla._compat import record_params
from scintilla.clustering._louvain_compat import import_louvain
from scintilla.config import DEFAULT_N_PCA_COMPS, RANDOM_SEED
from scintilla.dimensionality_reduction._common import get_representation, stand_in
from scintilla.io.loaders import ensure_anndata


def louvain_clustering(
    adata: Union[ad.AnnData, pd.DataFrame],
    resolution: float = 1.0,
    use_rep: str = "X_pca",
    random_state: int = RANDOM_SEED,
    *,
    key_added: Optional[str] = None,
) -> np.ndarray:
    """Run Louvain clustering via scanpy and return the cluster labels.

    The neighbour graph and the clustering are computed on a stand-in object that holds
    only the representation, so ``adata`` is not modified (no ``"louvain"`` column and no
    neighbour graph is left behind) unless ``key_added`` is given.

    Parameters
    ----------
    adata
        Annotated data matrix (AnnData preferred).
    resolution
        Louvain resolution parameter; higher gives more clusters.
    use_rep
        Key in ``adata.obsm`` of the representation used for the neighbour graph.
        When it is absent a temporary PCA with 30 components is used.
    random_state
        Random seed.
    key_added
        If given, also store the labels (as a categorical) in ``adata.obs[key_added]``.

    Returns
    -------
    numpy.ndarray
        Integer cluster labels, one per cell.
    """
    try:
        import scanpy as sc
    except ImportError as exc:
        raise ImportError(
            "scanpy is required for Louvain clustering. "
            "Install with: pip install scanpy"
        ) from exc

    try:
        import_louvain()
    except ImportError:
        pass  # scanpy will raise a more informative error if needed
    adata = ensure_anndata(adata)
    rep = get_representation(adata, use_rep, random_state, n_comps=DEFAULT_N_PCA_COMPS)
    tmp = stand_in(rep, adata.obs_names)
    sc.pp.neighbors(tmp, use_rep="rep", random_state=random_state)
    sc.tl.louvain(tmp, resolution=resolution, random_state=random_state)
    labels = tmp.obs["louvain"].astype(int).values
    if key_added is not None:
        adata.obs[key_added] = pd.Categorical(labels.astype(str))
        record_params(
            adata, key_added, method="louvain", resolution=resolution, use_rep=use_rep, random_state=random_state
        )
    return labels
