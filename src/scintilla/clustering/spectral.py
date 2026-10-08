"""Spectral clustering with optional grid search."""

from __future__ import annotations

import warnings
from typing import Dict, List, Optional, Tuple

import anndata as ad
import numpy as np
import pandas as pd
from sklearn.cluster import SpectralClustering
from sklearn.metrics import silhouette_score

from scintilla.config import RANDOM_SEED
from scintilla.io.loaders import ensure_anndata


def _as_matrix(adata) -> np.ndarray:
    """Float64 matrix of an AnnData/DataFrame/array (sparse stays sparse)."""
    if isinstance(adata, (pd.DataFrame, ad.AnnData)):
        X = ensure_anndata(adata).X
        if hasattr(X, "toarray"):
            return X.astype(np.float64)  # preserve sparsity
        return np.asarray(X, dtype=np.float64)
    return np.asarray(adata, dtype=np.float64)


def spectral_clustering(
    adata,
    n_clusters: Optional[int] = None,
    affinity: str = "rbf",
    n_clusters_range: Optional[List[int]] = None,
    random_state: int = RANDOM_SEED,
) -> Tuple[np.ndarray, object, Dict]:
    """Run Spectral Clustering for one value of ``n_clusters``.

    Parameters
    ----------
    adata
        Annotated data matrix, DataFrame or numpy array.
    n_clusters
        Number of clusters.
    affinity
        Affinity kernel (``"rbf"``, ``"nearest_neighbors"``, ...).
    n_clusters_range
        Deprecated, together with ``n_clusters=None``: a grid search returns a different
        shape, so use :func:`spectral_grid_search`.
    random_state
        Random seed.

    Returns
    -------
    labels : numpy.ndarray
    model : sklearn.cluster.SpectralClustering
    metrics : dict
        ``n_clusters`` and, when at least two clusters were found, ``silhouette``.
    """
    if n_clusters is None:
        warnings.warn(
            "spectral_clustering(n_clusters=None) runs a grid search and returns a different "
            "shape; this is deprecated, use spectral_grid_search().",
            DeprecationWarning,
            stacklevel=2,
        )
        return spectral_grid_search(  # type: ignore[return-value]
            adata, n_clusters_range=n_clusters_range, affinity=affinity, random_state=random_state,
        )

    X = _as_matrix(adata)
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


def spectral_grid_search(
    adata,
    n_clusters_range: Optional[List[int]] = None,
    affinity: str = "rbf",
    random_state: int = RANDOM_SEED,
) -> Tuple[pd.DataFrame, np.ndarray]:
    """Run Spectral Clustering over a grid of ``n_clusters`` and keep the best silhouette.

    Parameters
    ----------
    adata
        Annotated data matrix, DataFrame or numpy array.
    n_clusters_range
        Values to try; defaults to ``scintilla.config.SPECTRAL_N_CLUSTERS_RANGE``.
    affinity
        Affinity kernel.
    random_state
        Random seed.

    Returns
    -------
    results_df : pandas.DataFrame
        One row per grid point with ``silhouette``, ``status`` (``"ok"`` or ``"failed"``)
        and ``failure_reason``; failed grid points stay in the table.
    best_labels : numpy.ndarray
        Labels of the grid point with the highest silhouette.
    """
    from scintilla.config import SPECTRAL_N_CLUSTERS_RANGE

    X = _as_matrix(adata)
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

    return pd.DataFrame(records), best_labels
