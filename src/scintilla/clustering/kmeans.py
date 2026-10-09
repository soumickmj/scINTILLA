"""K-Means clustering variants."""

from __future__ import annotations

from typing import Dict, Optional, Tuple, Union

import anndata as ad
import numpy as np
import pandas as pd
from sklearn.cluster import BisectingKMeans, KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import normalize

from scintilla.clustering._common import resolve_matrix, store_labels
from scintilla.config import RANDOM_SEED


def kmeans_clustering(
    adata: Union[pd.DataFrame, ad.AnnData, np.ndarray],
    n_clusters: int,
    init: str = "k-means++",
    spherical: bool = False,
    bisecting: bool = False,
    random_state: int = RANDOM_SEED,
    *,
    use_rep: Optional[str] = None,
    key_added: Optional[str] = None,
) -> Tuple[np.ndarray, object, Dict]:
    """Run K-Means (or variants) and return labels, model, metrics.

    Parameters
    ----------
    adata:
        Input data matrix or AnnData.
    n_clusters:
        Number of clusters.
    init:
        Initialisation method ('k-means++' or 'random').
    spherical:
        If True, L2-normalise rows before clustering.
    bisecting:
        If True, use BisectingKMeans.
    random_state:
        Random seed.
    use_rep:
        Key in ``adata.obsm`` of the representation to cluster (AnnData input only);
        ``None`` clusters ``adata.X``.
    key_added:
        If given, also store the labels in ``adata.obs[key_added]`` (AnnData input only).

    Returns
    -------
    labels : np.ndarray
    model : fitted sklearn model
    metrics : dict with 'inertia' and 'silhouette'
    """
    X = resolve_matrix(adata, use_rep)

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
    store_labels(adata, labels, key_added, "kmeans", n_clusters=n_clusters, init=init, spherical=spherical,
                 bisecting=bisecting, use_rep=use_rep, random_state=random_state)
    return labels, model, metrics
