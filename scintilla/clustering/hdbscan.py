"""HDBSCAN clustering with parameter grid search."""

from __future__ import annotations

import warnings
from typing import List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from scintilla.config import RANDOM_SEED
from scintilla.io.loaders import ensure_anndata


def hdbscan_clustering(
    data,
    min_cluster_size_range: List[int] = None,
    min_samples_range: List[Optional[int]] = None,
    metric: str = "euclidean",
) -> Tuple[pd.DataFrame, np.ndarray]:
    """Run HDBSCAN over a parameter grid.

    Parameters
    ----------
    data:
        Input data matrix, AnnData, or numpy array.
    min_cluster_size_range:
        List of min_cluster_size values to try.
    min_samples_range:
        List of min_samples values to try.
    metric:
        Distance metric.

    Returns
    -------
    results_df : pd.DataFrame  columns=[min_cluster_size, min_samples, n_clusters, n_noise]
    best_labels : np.ndarray  labels from best parameter set (most clusters, least noise)
    """
    from scintilla.config import HDBSCAN_MIN_CLUSTER_SIZE_RANGE, HDBSCAN_MIN_SAMPLES_RANGE  # noqa: PLC0415

    if min_cluster_size_range is None:
        min_cluster_size_range = HDBSCAN_MIN_CLUSTER_SIZE_RANGE
    if min_samples_range is None:
        min_samples_range = HDBSCAN_MIN_SAMPLES_RANGE

    # Try hdbscan package first, then sklearn>=1.3
    _HDBSCAN = None
    try:
        import hdbscan as hdbscan_pkg  # noqa: PLC0415
        _HDBSCAN = hdbscan_pkg.HDBSCAN
        _use_sklearn = False
    except ImportError:
        try:
            from sklearn.cluster import HDBSCAN as SkHDBSCAN  # noqa: PLC0415
            _HDBSCAN = SkHDBSCAN
            _use_sklearn = True
        except ImportError as exc:
            raise ImportError(
                "HDBSCAN requires either the 'hdbscan' package or scikit-learn>=1.3. "
                "Install with: pip install hdbscan  or  pip install scikit-learn>=1.3"
            ) from exc

    import anndata as ad  # noqa: PLC0415

    if isinstance(data, (pd.DataFrame, ad.AnnData)):
        adata = ensure_anndata(data)
        X = adata.X if not hasattr(adata.X, "toarray") else adata.X.toarray()
        X = X.astype(np.float64)
    else:
        X = np.asarray(data, dtype=np.float64)

    records = []
    best_labels = np.full(X.shape[0], -1, dtype=int)
    best_score = -1

    for mcs in min_cluster_size_range:
        for ms in min_samples_range:
            try:
                if _use_sklearn:
                    kwargs = {"min_cluster_size": mcs, "metric": metric}
                    if ms is not None:
                        kwargs["min_samples"] = ms
                    model = _HDBSCAN(**kwargs)
                else:
                    kwargs = {"min_cluster_size": mcs, "metric": metric, "core_dist_n_jobs": -1}
                    if ms is not None:
                        kwargs["min_samples"] = ms
                    model = _HDBSCAN(**kwargs)
                labels = model.fit_predict(X)
                n_clusters = len(np.unique(labels[labels != -1]))
                n_noise = int((labels == -1).sum())
                records.append({
                    "min_cluster_size": mcs,
                    "min_samples": ms,
                    "n_clusters": n_clusters,
                    "n_noise": n_noise,
                    "status": "ok",
                })
                score = n_clusters - n_noise / (X.shape[0] + 1)
                if score > best_score:
                    best_score = score
                    best_labels = labels
            except MemoryError:
                raise
            except Exception as exc:
                warnings.warn(
                    "HDBSCAN grid point failed "
                    f"(min_cluster_size={mcs}, min_samples={ms}): {exc}",
                    UserWarning,
                    stacklevel=2,
                )
                records.append({
                    "min_cluster_size": mcs,
                    "min_samples": ms,
                    "n_clusters": np.nan,
                    "n_noise": np.nan,
                    "status": "failed",
                    "failure_reason": str(exc),
                })

    results_df = pd.DataFrame(records)
    return results_df, best_labels
