"""Helpers shared by the two-group differential-expression tests."""

from __future__ import annotations

from typing import Optional, Tuple

import anndata as ad
import numpy as np

from scintilla._compat import get_matrix


def two_group_matrices(
    adata: ad.AnnData,
    group_col: str,
    group1: str,
    group2: str,
    layer: Optional[str] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return the dense float64 expression of the cells of ``group1`` and ``group2``.

    Only the cells of the two groups are densified, so comparing two clusters of a large
    sparse dataset does not allocate a dense copy of the whole matrix.  Row order within
    each group follows ``adata``.
    """
    if group_col not in adata.obs.columns:
        raise KeyError(f"Column '{group_col}' not found in obs.")
    groups = adata.obs[group_col].to_numpy()
    mask1 = groups == group1
    mask2 = groups == group2
    if not mask1.any() or not mask2.any():
        raise ValueError(f"Groups '{group1}' or '{group2}' not found.")
    keep = mask1 | mask2
    sub = adata[keep]
    X = get_matrix(sub, layer, reason="per-gene tests need a dense matrix of the two groups", dtype=np.float64)
    in_group1 = mask1[keep]
    return X[in_group1], X[~in_group1]
