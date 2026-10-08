"""Force-directed graph layout."""

from __future__ import annotations

from typing import Optional

import anndata as ad

from scintilla._compat import finish, prepare, record_params
from scintilla.config import RANDOM_SEED
from scintilla.dimensionality_reduction._common import get_representation, stand_in


def run_force_directed(
    adata: ad.AnnData,
    use_rep: str = "X_pca",
    random_state: int = RANDOM_SEED,
    *,
    layer: Optional[str] = None,
    key_added: str = "X_draw_graph_fa",
    copy: bool = False,
) -> Optional[ad.AnnData]:
    """Run a ForceAtlas2 layout via scanpy and store it in ``adata.obsm[key_added]``.

    ForceAtlas2 needs the optional ``fa2-modified`` package; without it scanpy falls
    back to the Fruchterman-Reingold layout, and the layout actually used is recorded
    in ``adata.uns["scintilla"]["draw_graph"]``.

    Parameters
    ----------
    adata
        Annotated data matrix.
    use_rep
        Representation in ``adata.obsm`` used to build the neighbour graph.  When it
        is absent a temporary PCA of ``X`` is used (and not stored).
    random_state
        Random seed.
    layer
        Layer used for the temporary PCA when ``use_rep`` is absent.
    key_added
        Key of the layout in ``adata.obsm``.
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
    rep = get_representation(adata, use_rep, random_state, layer)
    tmp = stand_in(rep, adata.obs_names)
    sc.pp.neighbors(tmp, use_rep="rep", random_state=random_state)
    sc.tl.draw_graph(tmp, layout="fa", random_state=random_state)
    # scanpy falls back to the Fruchterman-Reingold layout ("fr") when fa2-modified is missing.
    produced = [k for k in tmp.obsm if str(k).startswith("X_draw_graph_")]
    layout = produced[0].removeprefix("X_draw_graph_")
    adata.obsm[key_added] = tmp.obsm[produced[0]]
    record_params(adata, key_added, layout=layout, use_rep=use_rep, random_state=random_state, key_added=key_added)
    return finish(adata, give_back)
