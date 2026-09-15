"""UMAP dimensionality reduction."""

from __future__ import annotations

import anndata as ad

from scintilla.config import RANDOM_SEED


def run_umap(
    adata: ad.AnnData,
    n_neighbors: int = 15,
    min_dist: float = 0.5,
    n_components: int = 2,
    use_rep: str = "X_pca",
    random_state: int = RANDOM_SEED,
) -> ad.AnnData:
    """Run UMAP embedding via scanpy.

    Parameters
    ----------
    adata:
        AnnData with use_rep in obsm.
    n_neighbors:
        Number of neighbours for kNN graph.
    min_dist:
        Minimum distance in UMAP embedding.
    n_components:
        Number of UMAP dimensions.
    use_rep:
        Representation to use.

    Returns
    -------
    AnnData with X_umap in obsm.
    """
    try:
        import scanpy as sc  # noqa: PLC0415
    except ImportError as exc:
        raise ImportError("scanpy is required. Install with: pip install scanpy") from exc

    adata = adata.copy()
    if use_rep not in adata.obsm:
        sc.tl.pca(adata, random_state=random_state)
        use_rep = "X_pca"

    sc.pp.neighbors(
        adata, n_neighbors=n_neighbors, use_rep=use_rep,
        random_state=random_state,
    )
    sc.tl.umap(
        adata, min_dist=min_dist, n_components=n_components,
        random_state=random_state,
    )
    return adata
