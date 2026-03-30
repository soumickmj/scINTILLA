"""Utility functions for clustering."""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np
from scipy.cluster.hierarchy import cophenet
from scipy.spatial.distance import pdist


def map_clusters_to_labels(
    cluster_labels: np.ndarray,
    true_labels: np.ndarray,
) -> Dict[int, object]:
    """Map cluster IDs to true-label categories by majority vote.

    Returns
    -------
    mapping : dict
        {cluster_id: most_common_true_label}
    """
    mapping: Dict[int, object] = {}
    for cid in np.unique(cluster_labels):
        mask = cluster_labels == cid
        if mask.sum() == 0:
            continue
        values, counts = np.unique(true_labels[mask], return_counts=True)
        mapping[cid] = values[np.argmax(counts)]
    return mapping


def cophenetic_correlation(Z: np.ndarray, X: np.ndarray) -> float:
    """Compute the cophenetic correlation coefficient (CPCC).

    Parameters
    ----------
    Z:
        Scipy linkage matrix.
    X:
        Original data matrix (n_samples x n_features).

    Returns
    -------
    float : CPCC value.
    """
    dist = pdist(X, metric="euclidean")
    c, _ = cophenet(Z, dist)
    return float(c)
