"""ComBat batch correction via scanpy."""

from __future__ import annotations

from typing import Optional

import anndata as ad
import pandas as pd

from scintilla._compat import finish, get_matrix, prepare, record_params


def combat_correct(
    adata: ad.AnnData,
    batch_key: str,
    *,
    layer: Optional[str] = None,
    key_added: str = "combat",
    copy: bool = False,
) -> Optional[ad.AnnData]:
    """Apply ComBat batch correction and store the result in ``adata.layers[key_added]``.

    ComBat needs a dense matrix, so a sparse ``X`` is densified once (and only the
    corrected layer is stored, ``X`` is left as it was).

    Parameters
    ----------
    adata
        Annotated data matrix.
    batch_key
        Column in ``adata.obs`` with batch labels.
    layer
        Layer to correct; ``None`` corrects ``adata.X``.
    key_added
        Name of the layer that receives the corrected values.
    copy
        Return a modified copy instead of modifying ``adata`` in place.

    Returns
    -------
    anndata.AnnData or None
        ``None`` when working in place; the modified copy when ``copy=True``.
    """
    try:
        import scanpy as sc
    except ImportError as exc:
        raise ImportError("scanpy is required. Install with: pip install scanpy") from exc

    adata, give_back = prepare(adata, copy=copy)
    if batch_key not in adata.obs.columns:
        raise KeyError(f"Column '{batch_key}' not found in obs.")
    X = get_matrix(adata, layer, reason="ComBat needs a dense matrix").copy()
    work = ad.AnnData(X=X, obs=pd.DataFrame({batch_key: adata.obs[batch_key].to_numpy()}, index=adata.obs_names))
    sc.pp.combat(work, key=batch_key)
    adata.layers[key_added] = work.X
    record_params(adata, "combat", batch_key=batch_key, layer=layer, key_added=key_added)
    return finish(adata, give_back)
