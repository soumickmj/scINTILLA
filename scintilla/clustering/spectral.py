"""Spectral clustering with optional grid search."""

from __future__ import annotations

import warnings
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.cluster import SpectralClustering
from sklearn.metrics import silhouette_score

from scintilla.config import RANDOM_SEED
from scintilla.io.loaders import ensure_anndata


def spectral_clustering(
    data,
    n_clusters: Optional[int] = None,
    affinity: str = "rbf",
    n_clusters_range: Optional[List[int]] = None,
    random_state: int = RANDOM_SEED,
) -> Union[Tuple[np.ndarray, object, Dict], Tuple[pd.DataFrame, np.ndarray]]:
    """Run Spectral Clustering, optionally over a grid of n_clusters.

    Parameters
    ----------
    data:
        Input data matrix, AnnData, or numpy array.
    n_clusters:
        Number of clusters for single run. If None and n_clusters_range given,
        runs a grid search.
    affinity:
        Affinity kernel ('rbf', 'nearest_neighbors', etc.).
    n_clusters_range:
        If provided and n_clusters is None, runs grid search over these values.

    Returns
    -------
    Single run: (labels, model, metrics)
    Grid search: (results_df, best_labels)
    """
    from scintilla.config import SPECTRAL_N_CLUSTERS_RANGE  # noqa: PLC0415
    import anndata as ad  # noqa: PLC0415

    if isinstance(data, (pd.DataFrame, ad.AnnData)):
        adata = ensure_anndata(data)
        X = adata.X
        if hasattr(X, "toarray"):
            X = X.astype(np.float64)  # preserve sparsity
        else:
            X = np.asarray(X, dtype=np.float64)
    else:
        X = np.asarray(data, dtype=np.float64)

    if n_clusters is not None:
        # Single run
        model = SpectralClustering(
            n_clusters=n_clusters, affinity=affinity, random_state=random_state
        )
        labels = model.fit_predict(X)
        unique = np.unique(labels[labels != -1])
        metrics: Dict = {"n_clusters": len(unique)}
        if len(unique) >= 2:
            try:
                metrics["silhouette"] = float(silhouette_score(X, labels))
            except ValueError as exc:
                metrics["silhouette"] = float("nan")
                warnings.warn(f"Spectral silhouette failed: {exc}", stacklevel=2)
        return labels, model, metrics

    # Grid search
    if n_clusters_range is None:
        n_clusters_range = SPECTRAL_N_CLUSTERS_RANGE

    records = []
    best_labels = np.zeros(X.shape[0], dtype=int)
    best_sil = -1.0

    for nc in n_clusters_range:
        try:
            model = SpectralClustering(
                n_clusters=nc, affinity=affinity, random_state=random_state
            )
            labels = model.fit_predict(X)
            sil = float("nan")
            if len(np.unique(labels)) >= 2:
                try:
                    sil = float(silhouette_score(X, labels))
                except ValueError as exc:
                    warnings.warn(f"Spectral silhouette failed: {exc}", stacklevel=2)
            records.append({
                "n_clusters": nc, "affinity": affinity, "silhouette": sil,
                "status": "ok", "failure_reason": None,
            })
            if not np.isnan(sil) and sil > best_sil:
                best_sil = sil
                best_labels = labels
        except MemoryError:
            raise
        except Exception as exc:
            warnings.warn(f"Spectral failed: {exc}", stacklevel=2)
            records.append({
                "n_clusters": nc, "affinity": affinity, "silhouette": float("nan"),
                "status": "failed", "failure_reason": str(exc),
            })

    results_df = pd.DataFrame(records)
    return results_df, best_labels
