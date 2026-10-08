"""Scanorama batch correction."""

from __future__ import annotations

from typing import Optional

import anndata as ad
import numpy as np
import pandas as pd

from scintilla._compat import finish, get_matrix, prepare, record_params
from scintilla.config import RANDOM_SEED


def scanorama_correct(
    adata: ad.AnnData,
    batch_key: str,
    random_state: int = RANDOM_SEED,
    *,
    layer: Optional[str] = None,
    key_added: str = "X_scanorama",
    copy: bool = False,
) -> Optional[ad.AnnData]:
    """Apply Scanorama and store the integrated embedding in ``adata.obsm[key_added]``.

    Parameters
    ----------
    adata
        Annotated data matrix.
    batch_key
        Column in ``adata.obs`` with batch labels.
    random_state
        Random seed for Scanorama integration.
    layer
        Layer to integrate; ``None`` uses ``adata.X``.
    key_added
        Key of the integrated embedding in ``adata.obsm``.
    copy
        Return a modified copy instead of modifying ``adata`` in place.

    Returns
    -------
    anndata.AnnData or None
        ``None`` when working in place; the modified copy when ``copy=True``.
    """
    try:
        import scanorama
    except ImportError as exc:
        raise ImportError(
            "scanorama is required. Install with: pip install scanorama"
        ) from exc

    adata, give_back = prepare(adata, copy=copy)
    if batch_key not in adata.obs.columns:
        raise KeyError(f"Column '{batch_key}' not found in obs.")
    X = get_matrix(adata, layer, dense=False)
    batch_values = adata.obs[batch_key].to_numpy()
    batches = pd.unique(batch_values).tolist()
    # Stand-ins carry only the matrix and the cell names, not the caller's obsm/layers.
    var = pd.DataFrame(index=adata.var_names)
    adatas = []
    for b in batches:
        mask = batch_values == b
        adatas.append(ad.AnnData(X=X[mask], obs=pd.DataFrame(index=adata.obs_names[mask]), var=var.copy()))

    corrected = scanorama.correct_scanpy(
        adatas, return_dimred=True, seed=random_state,
    )
    order = []
    for adata_b in adatas:
        order.extend(adata_b.obs_names.tolist())

    emb = np.vstack([c.obsm["X_scanorama"] for c in corrected])
    # Reorder to match adata.obs_names
    idx_map = {name: i for i, name in enumerate(order)}
    reorder = [idx_map[name] for name in adata.obs_names]
    adata.obsm[key_added] = emb[reorder]
    record_params(adata, "scanorama", batch_key=batch_key, random_state=random_state, layer=layer, key_added=key_added)
    return finish(adata, give_back)
