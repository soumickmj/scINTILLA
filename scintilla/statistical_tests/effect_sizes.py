"""Standardised effect-size measures for differential expression.

All functions are vectorised along the gene axis for efficient computation
on the full expression matrix.
"""

from __future__ import annotations

import numpy as np


def rank_biserial(U: np.ndarray, n1: int, n2: int) -> np.ndarray:
    """Rank-biserial correlation from the Mann-Whitney U statistic.

    ``r = 1 - 2U / (n1 * n2)``

    Parameters
    ----------
    U:
        U statistic(s) — scalar or 1-D array (one per gene).
    n1, n2:
        Sample sizes of the two groups.

    Returns
    -------
    np.ndarray of rank-biserial correlations in [-1, 1].
    """
    U = np.asarray(U, dtype=np.float64)
    return 1.0 - (2.0 * U) / (n1 * n2)


def cohens_d(x1: np.ndarray, x2: np.ndarray) -> np.ndarray:
    """Cohen's d (pooled-SD standardised mean difference).

    Parameters
    ----------
    x1, x2:
        2-D arrays of shape (n_cells, n_genes).

    Returns
    -------
    1-D array of Cohen's d values (one per gene).
    """
    x1 = np.asarray(x1, dtype=np.float64)
    x2 = np.asarray(x2, dtype=np.float64)
    n1, n2 = x1.shape[0], x2.shape[0]
    mean_diff = x1.mean(axis=0) - x2.mean(axis=0)
    var1 = x1.var(axis=0, ddof=1)
    var2 = x2.var(axis=0, ddof=1)
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    return mean_diff / (pooled_std + 1e-10)


def hedges_g(x1: np.ndarray, x2: np.ndarray) -> np.ndarray:
    """Hedges' g — Cohen's d with small-sample-bias correction.

    ``g = d * J``  where  ``J = 1 - 3 / (4*(n1+n2-2) - 1)``

    Parameters
    ----------
    x1, x2:
        2-D arrays of shape (n_cells, n_genes).

    Returns
    -------
    1-D array of Hedges' g values.
    """
    d = cohens_d(x1, x2)
    n1, n2 = x1.shape[0], x2.shape[0]
    df = n1 + n2 - 2
    J = 1.0 - 3.0 / (4.0 * df - 1.0) if df > 0 else 1.0
    return d * J


def cliffs_delta(x1: np.ndarray, x2: np.ndarray) -> np.ndarray:
    """Cliff's delta — non-parametric effect size.

    For moderate gene counts (≤ 5000), uses vectorised broadcasting.
    Falls back to a loop-based computation for larger gene sets to avoid
    excessive memory allocation.

    Parameters
    ----------
    x1, x2:
        2-D arrays of shape (n_cells, n_genes).

    Returns
    -------
    1-D array of Cliff's delta values in [-1, 1].
    """
    x1 = np.asarray(x1, dtype=np.float64)
    x2 = np.asarray(x2, dtype=np.float64)
    n_genes = x1.shape[1]
    n1, n2 = x1.shape[0], x2.shape[0]

    # Guard against excessive memory allocation *before* creating the
    # broadcast array.  The vectorised path creates an (n1, n_genes, n2)
    # float64 tensor; check estimated size first.
    if n_genes <= 5000 and n1 * n2 * n_genes <= 5e8:
        # Vectorised: (n1, n_genes, 1) vs (n2, n_genes, 1)
        diff = x1[:, :, None] - x2[:, :, None].transpose(2, 1, 0)
        # Use sign: +1 if x1 > x2, -1 if x1 < x2, 0 if equal
        gt = (diff > 0).sum(axis=(0, 2))
        lt = (diff < 0).sum(axis=(0, 2))
        return (gt - lt) / (n1 * n2)

    return _cliffs_delta_loop(x1, x2)


def _cliffs_delta_loop(x1: np.ndarray, x2: np.ndarray) -> np.ndarray:
    """Fallback gene-by-gene Cliff's delta for large matrices."""
    n1, n2 = x1.shape[0], x2.shape[0]
    n_genes = x1.shape[1]
    result = np.empty(n_genes, dtype=np.float64)
    for g in range(n_genes):
        a = x1[:, g]
        b = x2[:, g]
        diff = np.subtract.outer(a, b)
        result[g] = (np.sum(diff > 0) - np.sum(diff < 0)) / (n1 * n2)
    return result
