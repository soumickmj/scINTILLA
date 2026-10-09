"""Diffusion map dimensionality reduction."""

from __future__ import annotations

from typing import Optional

import anndata as ad

from scintilla._compat import finish, prepare, record_params
from scintilla.config import RANDOM_SEED
from scintilla.dimensionality_reduction._common import get_representation, stand_in


def run_diffusion_map(
    adata: ad.AnnData,
    n_comps: int = 10,
    use_rep: str = "X_pca",
    random_state: int = RANDOM_SEED,
    *,
    layer: Optional[str] = None,
    key_added: str = "X_diffmap",
    copy: bool = False,
) -> Optional[ad.AnnData]:
    """Run a diffusion map via scanpy and store it in ``adata.obsm[key_added]``.

    Parameters
    ----------
    adata
        Annotated data matrix.
    n_comps
        Number of diffusion components.
    use_rep
        Representation in ``adata.obsm`` used to build the neighbour graph.  When it
        is absent a temporary PCA of ``X`` is used (and not stored).
    random_state
        Random seed.
    layer
        Layer used for the temporary PCA when ``use_rep`` is absent.
    key_added
        Key of the embedding in ``adata.obsm``.
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
    sc.tl.diffmap(tmp, n_comps=n_comps, random_state=random_state)
    adata.obsm[key_added] = tmp.obsm["X_diffmap"]
    record_params(
        adata, key_added, n_comps=n_comps, use_rep=use_rep, random_state=random_state, key_added=key_added
    )
    return finish(adata, give_back)
