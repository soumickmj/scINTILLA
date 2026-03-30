"""Embedding quality metrics."""

from __future__ import annotations

import numpy as np


def trustworthiness(X_high: np.ndarray, X_low: np.ndarray, n_neighbors: int = 10) -> float:
    """Trustworthiness of a low-dimensional embedding.

    Parameters
    ----------
    X_high:
        High-dimensional data.
    X_low:
        Low-dimensional embedding.
    n_neighbors:
        Number of neighbours.

    Returns
    -------
    float in [0, 1].
    """
    from sklearn.manifold import trustworthiness as _tw  # noqa: PLC0415

    k = min(n_neighbors, X_high.shape[0] - 1)
    return float(_tw(X_high, X_low, n_neighbors=k))


def knn_preservation(X_high: np.ndarray, X_low: np.ndarray, k: int = 10) -> float:
    """Fraction of k-NN preserved in low-dimensional embedding.

    Parameters
    ----------
    X_high:
        High-dimensional data.
    X_low:
        Low-dimensional embedding.
    k:
        Number of neighbours.

    Returns
    -------
    float in [0, 1].
    """
    from sklearn.neighbors import NearestNeighbors  # noqa: PLC0415

    n = X_high.shape[0]
    k = min(k, n - 1)

    nn_high = NearestNeighbors(n_neighbors=k).fit(X_high)
    nn_low = NearestNeighbors(n_neighbors=k).fit(X_low)

    _, idx_high = nn_high.kneighbors(X_high)
    _, idx_low = nn_low.kneighbors(X_low)

    overlap = 0
    for i in range(n):
        overlap += len(set(idx_high[i]) & set(idx_low[i]))

    return float(overlap / (n * k))
