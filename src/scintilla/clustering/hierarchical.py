"""Hierarchical clustering (sklearn fast mode + scipy diagnostics mode)."""

from __future__ import annotations

import warnings
from typing import Optional, Tuple, Union

import anndata as ad
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import pdist
from sklearn.cluster import AgglomerativeClustering

from scintilla.clustering._common import resolve_matrix, store_labels
from scintilla.clustering.utils import cophenetic_correlation


def hierarchical_sklearn(
    data: np.ndarray,
    n_clusters: int,
    metric: str = "euclidean",
    linkage_method: str = "complete",
) -> np.ndarray:
    """Fast sklearn-based agglomerative clustering (for benchmarking).

    Parameters
    ----------
    data
        Feature matrix (cells by features).
    n_clusters
        Number of clusters.
    metric
        Distance metric.
    linkage_method
        Linkage criterion (``ward`` requires the euclidean metric).
    """
    # Ward requires euclidean
    if linkage_method == "ward" and metric != "euclidean":
        raise ValueError("Ward linkage requires euclidean metric.")
    agg = AgglomerativeClustering(
        n_clusters=n_clusters,
        metric=metric,
        linkage=linkage_method,
    )
    return agg.fit_predict(data)


def hierarchical_scipy(
    data: np.ndarray,
    metric: str = "euclidean",
    linkage_method: str = "complete",
    n_clusters: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """Scipy-based hierarchical clustering with the cophenetic correlation (CPCC).

    Parameters
    ----------
    data
        Feature matrix (cells by features).
    metric
        Distance metric passed to :func:`scipy.spatial.distance.pdist`.
    linkage_method
        Linkage method passed to :func:`scipy.cluster.hierarchy.linkage`.
    n_clusters
        Cut the tree into this many clusters; if ``None`` the tree is cut at distance 0.7.

    Returns
    -------
    labels : numpy.ndarray
    Z : numpy.ndarray
        The linkage matrix; draw it with :func:`scintilla.pl.dendrogram`.
    cpcc : float
        Cophenetic correlation coefficient.
    """
    dist_vec = pdist(data, metric=metric)
    Z = linkage(dist_vec, method=linkage_method)
    cpcc = cophenetic_correlation(Z, data)

    if n_clusters is not None:
        labels = fcluster(Z, n_clusters, criterion="maxclust") - 1
    else:
        labels = fcluster(Z, t=0.7, criterion="distance")

    return labels, Z, cpcc


def hierarchical_clustering(
    adata: Union[ad.AnnData, pd.DataFrame, np.ndarray],
    n_clusters: int,
    metric: str = "euclidean",
    linkage: str = "complete",
    mode: str = "sklearn",
    *,
    use_rep: Optional[str] = None,
    key_added: Optional[str] = None,
) -> Union[np.ndarray, Tuple]:
    """Hierarchical (agglomerative) clustering; returns the cluster labels.

    Parameters
    ----------
    adata
        Annotated data matrix, DataFrame or array.
    n_clusters
        Number of clusters.
    metric
        Distance metric.
    linkage
        Linkage method.
    use_rep
        Key in ``adata.obsm`` of the representation to cluster (AnnData input only);
        ``None`` clusters ``adata.X``.
    key_added
        If given, also store the labels in ``adata.obs[key_added]`` (AnnData input only).
    mode
        ``"sklearn"`` (default) returns the labels.  ``"scipy"`` is deprecated because it
        changes the return type to ``(labels, Z, cpcc)``; call :func:`hierarchical_scipy`
        instead.

    Returns
    -------
    numpy.ndarray
        Cluster labels (``mode="sklearn"``).
    """
    X = resolve_matrix(adata, use_rep, dense=True)

    if mode == "scipy":
        warnings.warn(
            "hierarchical_clustering(mode='scipy') returns a tuple and is deprecated; "
            "call hierarchical_scipy() instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return hierarchical_scipy(X, metric=metric, linkage_method=linkage, n_clusters=n_clusters)
    labels = hierarchical_sklearn(X, n_clusters=n_clusters, metric=metric, linkage_method=linkage)
    store_labels(adata, labels, key_added, "hierarchical", n_clusters=n_clusters, metric=metric, linkage=linkage,
                 use_rep=use_rep)
    return labels
