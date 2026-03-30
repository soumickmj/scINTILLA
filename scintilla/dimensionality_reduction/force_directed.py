"""Force-directed graph layout."""

from __future__ import annotations

import anndata as ad


def run_force_directed(
    adata: ad.AnnData,
    use_rep: str = "X_pca",
) -> ad.AnnData:
    """Run force-directed graph layout (ForceAtlas2) via scanpy.

    Parameters
    ----------
    adata:
        AnnData.
    use_rep:
        Representation to use for neighbour graph.

    Returns
    -------
    AnnData with X_draw_graph_fa in obsm.
    """
    try:
        import scanpy as sc  # noqa: PLC0415
    except ImportError as exc:
        raise ImportError("scanpy is required. Install with: pip install scanpy") from exc

    adata = adata.copy()
    if use_rep not in adata.obsm:
        sc.tl.pca(adata)
        use_rep = "X_pca"

    sc.pp.neighbors(adata, use_rep=use_rep)
    sc.tl.draw_graph(adata, layout="fa")
    return adata
