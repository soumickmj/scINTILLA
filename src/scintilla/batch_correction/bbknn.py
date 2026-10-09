"""BBKNN batch correction."""

from __future__ import annotations

from typing import Optional

import anndata as ad

from scintilla._compat import finish, prepare, record_params
from scintilla.config import RANDOM_SEED


def bbknn_correct(
    adata: ad.AnnData,
    batch_key: str,
    n_pcs: int = 30,
    random_state: int = RANDOM_SEED,
    *,
    copy: bool = False,
) -> Optional[ad.AnnData]:
    """Apply BBKNN (Batch Balanced kNN) batch correction.

    BBKNN corrects the neighbour *graph*, not an embedding, so its result is
    ``adata.obsp["connectivities"]`` and ``adata.uns["neighbors"]`` (the keys are
    fixed by the ``bbknn`` package).  BBKNN needs ``adata.obsm["X_pca"]``; when it is
    missing it is computed and stored.

    Parameters
    ----------
    adata
        Annotated data matrix.
    batch_key
        Column in ``adata.obs`` with batch labels.
    n_pcs
        Number of principal components to use.
    random_state
        Seed for the PCA computed here when ``X_pca`` is missing, and for BBKNN's
        neighbour search.  BBKNN only consumes the seed when it runs with
        ``computation="pynndescent"``; under its default ``annoy`` backend the search
        is already deterministic and the seed is unused.
    copy
        Return a modified copy instead of modifying ``adata`` in place.

    Returns
    -------
    anndata.AnnData or None
        ``None`` when working in place; the modified copy when ``copy=True``.
    """
    try:
        import bbknn
    except ImportError as exc:
        raise ImportError(
            "bbknn is required. Install with: pip install bbknn"
        ) from exc

    from scintilla.preprocessing.pca import run_pca

    adata, give_back = prepare(adata, copy=copy)
    if batch_key not in adata.obs.columns:
        raise KeyError(f"Column '{batch_key}' not found in obs.")
    if "X_pca" not in adata.obsm:
        run_pca(adata, n_comps=n_pcs, random_state=random_state)

    bbknn.bbknn(
        adata,
        batch_key=batch_key,
        n_pcs=n_pcs,
        pynndescent_random_state=random_state,
    )
    record_params(adata, "bbknn", batch_key=batch_key, n_pcs=n_pcs, random_state=random_state)
    return finish(adata, give_back)
