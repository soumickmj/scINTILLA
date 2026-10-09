"""Helpers shared by the embedding functions.

The embedding functions work on a lightweight stand-in AnnData (just the
representation), so the neighbour graph, PCA and scanpy bookkeeping that they
need never leak into the caller's object: only ``obsm[key_added]`` is written.
"""

from __future__ import annotations

from typing import Optional

import anndata as ad
import numpy as np
import pandas as pd

from scintilla._logging import logger


def get_representation(
    adata: ad.AnnData,
    use_rep: str,
    random_state: int,
    layer: Optional[str] = None,
    n_comps: int = 50,
) -> np.ndarray:
    """Return ``adata.obsm[use_rep]``, or a throw-away PCA of ``X`` when it is absent.

    ``n_comps`` is capped at ``min(n_obs, n_vars) - 1``.

    Parameters
    ----------
    adata
        Annotated data matrix.
    use_rep
        Key in ``adata.obsm`` of the representation to use.
    random_state
        Random seed.
    layer
        Layer to use; ``None`` uses ``adata.X``.
    n_comps
        Number of components of the temporary PCA.
    """
    if use_rep in adata.obsm:
        return np.asarray(adata.obsm[use_rep])
    import scanpy as sc

    logger.info("obsm[%r] not found; computing a temporary PCA (not stored)", use_rep)
    X = adata.X if layer is None else adata.layers[layer]
    tmp = ad.AnnData(X=X, obs=pd.DataFrame(index=adata.obs_names))
    sc.pp.pca(tmp, n_comps=max(1, min(n_comps, min(adata.shape) - 1)), random_state=random_state)
    return np.asarray(tmp.obsm["X_pca"])


def stand_in(rep: np.ndarray, obs_names: pd.Index) -> ad.AnnData:
    """Build a minimal AnnData whose only content is the representation ``rep``.

    Parameters
    ----------
    rep
        Representation (cells by dimensions).
    obs_names
        Cell names.
    """
    tmp = ad.AnnData(
        X=np.zeros((rep.shape[0], 1), dtype=np.float32),
        obs=pd.DataFrame(index=obs_names),
    )
    tmp.obsm["rep"] = rep
    return tmp
