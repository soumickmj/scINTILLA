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

from scintilla._compat import get_matrix
from scintilla.clustering.utils import cophenetic_correlation
from scintilla.io.loaders import ensure_anndata


def hierarchical_sklearn(
    data: np.ndarray,
    n_clusters: int,
    metric: str = "euclidean",
    linkage_method: str = "complete",
) -> np.ndarray:
    """Fast sklearn-based agglomerative clustering (for benchmarking)."""
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
    mode
        ``"sklearn"`` (default) returns the labels.  ``"scipy"`` is deprecated because it
        changes the return type to ``(labels, Z, cpcc)``; call :func:`hierarchical_scipy`
        instead.

    Returns
    -------
    numpy.ndarray
        Cluster labels (``mode="sklearn"``).
    """
    if isinstance(adata, (pd.DataFrame, ad.AnnData)):
        adata = ensure_anndata(adata)
        X = get_matrix(adata, reason="hierarchical clustering needs a dense matrix")
        X = X.astype(np.float64)
    else:
        X = np.asarray(adata, dtype=np.float64)

    if mode == "scipy":
        warnings.warn(
            "hierarchical_clustering(mode='scipy') returns a tuple and is deprecated; "
            "call hierarchical_scipy() instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return hierarchical_scipy(X, metric=metric, linkage_method=linkage, n_clusters=n_clusters)
    return hierarchical_sklearn(X, n_clusters=n_clusters, metric=metric, linkage_method=linkage)
