"""BBKNN batch correction."""

from __future__ import annotations

import anndata as ad

from scintilla.config import RANDOM_SEED


def bbknn_correct(
    adata: ad.AnnData,
    batch_key: str,
    n_pcs: int = 30,
    random_state: int = RANDOM_SEED,
) -> ad.AnnData:
    """Apply BBKNN (Batch Balanced kNN) batch correction.

    Parameters
    ----------
    adata:
        AnnData with X_pca in obsm.
    batch_key:
        Column in obs with batch labels.
    n_pcs:
        Number of PCs to use.
    random_state:
        Random seed for the PCA computed here when ``X_pca`` is missing, and
        for BBKNN's neighbour search.  BBKNN only consumes the seed when it
        runs with ``computation="pynndescent"``; under its default ``annoy``
        backend the search is already deterministic and the seed is unused.

    Returns
    -------
    AnnData with BBKNN connectivities.
    """
    try:
        import bbknn  # noqa: PLC0415
    except ImportError as exc:
        raise ImportError(
            "bbknn is required. Install with: pip install bbknn"
        ) from exc

    import scanpy as sc  # noqa: PLC0415

    adata = adata.copy()
    if "X_pca" not in adata.obsm:
        sc.tl.pca(adata, n_comps=n_pcs, random_state=random_state)

    bbknn.bbknn(
        adata,
        batch_key=batch_key,
        n_pcs=n_pcs,
        pynndescent_random_state=random_state,
    )
    return adata
