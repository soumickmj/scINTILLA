"""UMAP dimensionality reduction."""

from __future__ import annotations

from typing import Optional

import anndata as ad

from scintilla._compat import finish, prepare, record_params
from scintilla.config import RANDOM_SEED
from scintilla.dimensionality_reduction._common import get_representation, stand_in


def run_umap(
    adata: ad.AnnData,
    n_neighbors: int = 15,
    min_dist: float = 0.5,
    n_components: int = 2,
    use_rep: str = "X_pca",
    random_state: int = RANDOM_SEED,
    *,
    layer: Optional[str] = None,
    key_added: str = "X_umap",
    copy: bool = False,
) -> Optional[ad.AnnData]:
    """Run UMAP and store the embedding in ``adata.obsm[key_added]``.

    The neighbour graph is built from ``adata.obsm[use_rep]`` on a stand-in object, so
    nothing but the embedding is written to ``adata``.

    Parameters
    ----------
    adata
        Annotated data matrix.
    n_neighbors
        Number of neighbours for the kNN graph.
    min_dist
        Minimum distance in the UMAP embedding.
    n_components
        Number of UMAP dimensions.
    use_rep
        Representation in ``adata.obsm`` to embed.  When it is absent a temporary PCA
        of ``X`` is used (and not stored).
    random_state
        Seed for the neighbour search and the embedding.
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
    sc.pp.neighbors(tmp, n_neighbors=n_neighbors, use_rep="rep", random_state=random_state)
    sc.tl.umap(tmp, min_dist=min_dist, n_components=n_components, random_state=random_state)
    adata.obsm[key_added] = tmp.obsm["X_umap"]
    record_params(
        adata, key_added, n_neighbors=n_neighbors, min_dist=min_dist, n_components=n_components,
        use_rep=use_rep, random_state=random_state, key_added=key_added,
    )
    return finish(adata, give_back)
