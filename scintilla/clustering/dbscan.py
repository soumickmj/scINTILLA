"""DBSCAN clustering."""

from __future__ import annotations

from typing import Tuple, Union

import anndata as ad
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score

from scintilla.io.loaders import ensure_anndata


def dbscan_clustering(
    data: Union[pd.DataFrame, ad.AnnData, np.ndarray],
    eps: float = 0.5,
    min_samples: int = 5,
    metric: str = "euclidean",
) -> Tuple[np.ndarray, int, int, float]:
    """Run DBSCAN clustering.

    Returns
    -------
    labels : np.ndarray  (-1 = noise)
    n_clusters : int
    n_noise : int
    silhouette : float  (NaN if fewer than 2 clusters or all noise)
    """
    if isinstance(data, (pd.DataFrame, ad.AnnData)):
        adata = ensure_anndata(data)
        X = adata.X
        if hasattr(X, "toarray"):
            X = X.astype(np.float64)  # preserve sparsity
        else:
            X = np.asarray(X, dtype=np.float64)
    else:
        X = np.asarray(data, dtype=np.float64)

    db = DBSCAN(eps=eps, min_samples=min_samples, metric=metric)
    labels = db.fit_predict(X)

    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = int(np.sum(labels == -1))

    # Silhouette on non-noise points
    mask = labels != -1
    if n_clusters >= 2 and mask.sum() >= 2:
        sil = float(silhouette_score(X[mask], labels[mask]))
    else:
        sil = float("nan")

    return labels, n_clusters, n_noise, sil


def estimate_eps(
    data: Union[pd.DataFrame, ad.AnnData, np.ndarray],
    min_samples: int = 5,
) -> float:
    """Data-driven eps estimation via the k-distance graph elbow method.

    Computes the k-nearest-neighbour distances for all points (k = *min_samples*),
    sorts them, and identifies the "elbow" using the Kneedle algorithm (or a
    maximum-curvature fallback).

    Parameters
    ----------
    data:
        Feature matrix or AnnData.
    min_samples:
        DBSCAN ``min_samples`` parameter (used as *k* for the k-distance
        graph).

    Returns
    -------
    float — estimated eps value.
    """
    from sklearn.neighbors import NearestNeighbors  # noqa: PLC0415
    from scintilla.statistical_tests.adaptive import kneedle_elbow  # noqa: PLC0415

    if isinstance(data, (pd.DataFrame, ad.AnnData)):
        adata = ensure_anndata(data)
        X = adata.X
        if hasattr(X, "toarray"):
            X = X.toarray()
        X = np.asarray(X, dtype=np.float64)
    else:
        X = np.asarray(data, dtype=np.float64)

    k = min(min_samples, X.shape[0] - 1)
    nn = NearestNeighbors(n_neighbors=k).fit(X)
    distances, _ = nn.kneighbors(X)
    k_distances = np.sort(distances[:, -1])

    x_arr = np.arange(len(k_distances), dtype=np.float64)
    elbow_idx = kneedle_elbow(x_arr, k_distances, direction="increasing")
    return float(k_distances[elbow_idx])
