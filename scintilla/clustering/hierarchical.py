"""Hierarchical clustering (sklearn fast mode + scipy diagnostics mode)."""

from __future__ import annotations

from typing import Optional, Tuple, Union

import anndata as ad
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram
from scipy.spatial.distance import pdist
from sklearn.cluster import AgglomerativeClustering

from scintilla.io.loaders import ensure_anndata
from scintilla.clustering.utils import cophenetic_correlation

import matplotlib.pyplot as plt


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
) -> Tuple[np.ndarray, np.ndarray, float, plt.Figure]:
    """Scipy-based hierarchical clustering with dendrogram and CPCC.

    Returns
    -------
    labels, Z, cpcc, fig_dendrogram
    """
    dist_vec = pdist(data, metric=metric)
    Z = linkage(dist_vec, method=linkage_method)
    cpcc = cophenetic_correlation(Z, data)

    if n_clusters is not None:
        labels = fcluster(Z, n_clusters, criterion="maxclust") - 1
    else:
        labels = fcluster(Z, t=0.7, criterion="distance")

    fig, ax = plt.subplots(figsize=(10, 5))
    dendrogram(Z, ax=ax, no_labels=True, truncate_mode="lastp", p=30)
    ax.set_title(f"Dendrogram ({metric} / {linkage_method}), CPCC={cpcc:.3f}")
    plt.tight_layout()

    return labels, Z, cpcc, fig


def hierarchical_clustering(
    data: Union[pd.DataFrame, ad.AnnData, np.ndarray],
    n_clusters: int,
    metric: str = "euclidean",
    linkage: str = "complete",
    mode: str = "sklearn",
) -> Union[np.ndarray, Tuple]:
    """Dispatcher for hierarchical clustering.

    Parameters
    ----------
    mode:
        'sklearn' (fast) or 'scipy' (detailed with CPCC + dendrogram).
    """
    if isinstance(data, (pd.DataFrame, ad.AnnData)):
        adata = ensure_anndata(data)
        X = adata.X if not hasattr(adata.X, "toarray") else adata.X.toarray()
        X = X.astype(np.float64)
    else:
        X = np.asarray(data, dtype=np.float64)

    if mode == "scipy":
        return hierarchical_scipy(X, metric=metric, linkage_method=linkage, n_clusters=n_clusters)
    else:
        return hierarchical_sklearn(X, n_clusters=n_clusters, metric=metric, linkage_method=linkage)
