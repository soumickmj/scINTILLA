"""K-Means clustering variants."""

from __future__ import annotations

from typing import Dict, Optional, Tuple, Union

import anndata as ad
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, BisectingKMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import normalize

from scintilla.config import RANDOM_SEED
from scintilla.io.loaders import ensure_anndata


def kmeans_clustering(
    data: Union[pd.DataFrame, ad.AnnData, np.ndarray],
    n_clusters: int,
    init: str = "k-means++",
    spherical: bool = False,
    bisecting: bool = False,
    random_state: int = RANDOM_SEED,
) -> Tuple[np.ndarray, object, Dict]:
    """Run K-Means (or variants) and return labels, model, metrics.

    Parameters
    ----------
    data:
        Input data matrix or AnnData.
    n_clusters:
        Number of clusters.
    init:
        Initialisation method ('k-means++' or 'random').
    spherical:
        If True, L2-normalise rows before clustering.
    bisecting:
        If True, use BisectingKMeans.

    Returns
    -------
    labels : np.ndarray
    model : fitted sklearn model
    metrics : dict with 'inertia' and 'silhouette'
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

    if spherical:
        X = normalize(X, norm="l2")

    if bisecting:
        model = BisectingKMeans(
            n_clusters=n_clusters,
            init=init,
            random_state=random_state,
            n_init=3,
        )
    else:
        model = KMeans(
            n_clusters=n_clusters,
            init=init,
            random_state=random_state,
            n_init=10,
        )

    labels = model.fit_predict(X)
    inertia = float(model.inertia_) if hasattr(model, "inertia_") else np.nan
    sil = float(silhouette_score(X, labels)) if len(np.unique(labels)) >= 2 else np.nan

    metrics = {"inertia": inertia, "silhouette": sil}
    return labels, model, metrics
