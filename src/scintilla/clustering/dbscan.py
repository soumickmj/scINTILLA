"""DBSCAN clustering."""

from __future__ import annotations

from typing import Optional, Tuple, Union

import anndata as ad
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score

from scintilla.clustering._common import resolve_matrix, store_labels


def dbscan_clustering(
    adata: Union[pd.DataFrame, ad.AnnData, np.ndarray],
    eps: float = 0.5,
    min_samples: int = 5,
    metric: str = "euclidean",
    *,
    use_rep: Optional[str] = None,
    key_added: Optional[str] = None,
) -> Tuple[np.ndarray, int, int, float]:
    """Run DBSCAN clustering.

    Parameters
    ----------
    adata
        Annotated data matrix.
    eps
        Neighbourhood radius.
    min_samples
        Minimum number of points in a neighbourhood for a core point.
    metric
        Distance metric.
    use_rep
        Key in ``adata.obsm`` of the representation to cluster (AnnData input only);
        ``None`` clusters ``adata.X``.
    key_added
        If given, also store the labels in ``adata.obs[key_added]`` (AnnData input only).

    Returns
    -------
    labels : np.ndarray  (-1 = noise)
    n_clusters : int
    n_noise : int
    silhouette : float  (NaN if fewer than 2 clusters or all noise)
    """
    X = resolve_matrix(adata, use_rep)

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

    store_labels(adata, labels, key_added, "dbscan", eps=eps, min_samples=min_samples, metric=metric, use_rep=use_rep)
    return labels, n_clusters, n_noise, sil


def estimate_eps(
    adata: Union[pd.DataFrame, ad.AnnData, np.ndarray],
    min_samples: int = 5,
    *,
    use_rep: Optional[str] = None,
) -> float:
    """Data-driven eps estimation via the k-distance graph elbow method.

    Computes the k-nearest-neighbour distances for all points (k = *min_samples*),
    sorts them, and identifies the "elbow" using the Kneedle algorithm (or a
    maximum-curvature fallback).

    Parameters
    ----------
    adata
        Feature matrix or AnnData.
    min_samples
        DBSCAN ``min_samples`` parameter (used as *k* for the k-distance
        graph).
    use_rep
        Key in ``adata.obsm`` of the representation to use (AnnData input only);
        ``None`` uses ``adata.X``.

    Returns
    -------
    float — estimated eps value.
    """
    from sklearn.neighbors import NearestNeighbors

    from scintilla.statistical_tests.adaptive import kneedle_elbow

    X = resolve_matrix(adata, use_rep, dense=True)

    k = min(min_samples, X.shape[0] - 1)
    nn = NearestNeighbors(n_neighbors=k).fit(X)
    distances, _ = nn.kneighbors(X)
    k_distances = np.sort(distances[:, -1])

    x_arr = np.arange(len(k_distances), dtype=np.float64)
    elbow_idx = kneedle_elbow(x_arr, k_distances, direction="increasing")
    return float(k_distances[elbow_idx])
