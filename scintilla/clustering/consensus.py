"""Consensus clustering across multiple methods."""

from __future__ import annotations

import warnings
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage

from scintilla.config import RANDOM_SEED, LEIDEN_RESOLUTIONS
from scintilla.io.loaders import ensure_anndata

SUPPORTED_CONSENSUS_METHODS = ("kmeans", "leiden", "spectral")

# Reserved key in the returned stability mapping.  Method names are validated
# against SUPPORTED_CONSENSUS_METHODS, so this can never shadow a real method.
_FAILURES_KEY = "failures"


def consensus_clustering(
    adata,
    methods: Optional[List[str]] = None,
    n_runs_per_method: int = 5,
    resolution_range: Optional[List[float]] = None,
    random_state: int = RANDOM_SEED,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Union[float, Dict[str, str]]]]:
    """Consensus clustering by aggregating multiple runs.

    Runs multiple clustering methods with varied hyperparameters and builds a
    co-occurrence (consensus) matrix.  Hierarchical clustering of the consensus
    matrix yields stable final labels.

    Parameters
    ----------
    adata:
        AnnData object.
    methods:
        Clustering methods to use ('kmeans', 'leiden', 'spectral').
        Defaults to ['kmeans'].  Unknown names raise ``ValueError`` rather than
        contributing nothing to the consensus matrix.
    n_runs_per_method:
        Number of parameter variations per method.
    resolution_range:
        Leiden resolution values (used if 'leiden' in methods).
    random_state:
        Seed forwarded to every clustering run.

    Returns
    -------
    consensus_matrix : np.ndarray  (n_cells x n_cells) co-clustering frequency in [0,1]
    consensus_labels : np.ndarray  final labels from hierarchical cut
    stability_scores : dict
        One ``{method: float}`` entry per requested method, holding the mean
        pairwise ARI between that method's runs (``nan`` when fewer than two
        runs succeeded).  When any run failed, the reserved key ``"failures"``
        additionally maps ``{method: first_error_message}``; it is absent when
        every run succeeded.  Method names are validated, so ``"failures"``
        never collides with a method entry.
    """
    import anndata as ad  # noqa: PLC0415
    from scintilla.clustering.kmeans import kmeans_clustering  # noqa: PLC0415

    if methods is None:
        methods = ["kmeans"]
    unknown = [m for m in methods if m not in SUPPORTED_CONSENSUS_METHODS]
    if unknown:
        raise ValueError(
            f"Unknown consensus method(s) {unknown}; expected any of "
            f"{list(SUPPORTED_CONSENSUS_METHODS)}"
        )
    if resolution_range is None:
        resolution_range = LEIDEN_RESOLUTIONS[:n_runs_per_method]

    if isinstance(adata, (pd.DataFrame,)):
        adata = ensure_anndata(adata)

    X = adata.X if not hasattr(adata.X, "toarray") else adata.X.toarray()
    X = X.astype(np.float64)
    n_cells = X.shape[0]

    co_matrix = np.zeros((n_cells, n_cells), dtype=np.float64)
    n_total_runs = 0
    stability_scores: Dict[str, Union[float, Dict[str, str]]] = {}
    failures: Dict[str, str] = {}

    for method in methods:
        method_labels_list = []
        method_failures = []
        if method == "kmeans":
            k_values = list(range(2, n_runs_per_method + 2))
            for k in k_values:
                try:
                    labels, _, _ = kmeans_clustering(
                        X, n_clusters=k, random_state=random_state,
                    )
                    method_labels_list.append(labels)
                except MemoryError:
                    raise
                except Exception as exc:
                    method_failures.append(str(exc))
        elif method == "leiden":
            try:
                from scintilla.clustering.leiden import leiden_clustering  # noqa: PLC0415
                resolutions = resolution_range[:n_runs_per_method]
                for res in resolutions:
                    try:
                        labels = leiden_clustering(
                            adata, resolution=res, random_state=random_state,
                        )
                        method_labels_list.append(labels)
                    except MemoryError:
                        raise
                    except Exception as exc:
                        method_failures.append(str(exc))
            except ImportError as exc:
                method_failures.append(str(exc))
        elif method == "spectral":
            from scintilla.clustering.spectral import spectral_clustering  # noqa: PLC0415
            k_values = list(range(2, n_runs_per_method + 2))
            for k in k_values:
                try:
                    labels, _, _ = spectral_clustering(
                        X, n_clusters=k, random_state=random_state,
                    )
                    method_labels_list.append(labels)
                except MemoryError:
                    raise
                except Exception as exc:
                    method_failures.append(str(exc))

        # Build co-matrix from this method's runs
        method_co = np.zeros((n_cells, n_cells), dtype=np.float64)
        for labels in method_labels_list:
            for c in np.unique(labels[labels != -1]):
                idx = np.where(labels == c)[0]
                method_co[np.ix_(idx, idx)] += 1.0
        if method_labels_list:
            method_co /= len(method_labels_list)
            co_matrix += method_co
            n_total_runs += 1
        if method_failures:
            failures[method] = method_failures[0]
            warnings.warn(f"{method} failed: {method_failures[0]}", stacklevel=2)

        # Stability: avg pairwise label agreement within method
        if len(method_labels_list) >= 2:
            agreements = []
            stability_error = None
            for i in range(len(method_labels_list)):
                for j in range(i + 1, len(method_labels_list)):
                    try:
                        from sklearn.metrics import adjusted_rand_score  # noqa: PLC0415
                        agreements.append(adjusted_rand_score(method_labels_list[i], method_labels_list[j]))
                    except (ValueError, ArithmeticError) as exc:
                        stability_error = stability_error or exc
            if stability_error is not None:
                warnings.warn(
                    f"{method} stability score failed: {stability_error}",
                    stacklevel=2,
                )
            stability_scores[method] = float(np.mean(agreements)) if agreements else float("nan")
        else:
            stability_scores[method] = float("nan")

    if failures:
        stability_scores[_FAILURES_KEY] = failures

    if n_total_runs > 0:
        co_matrix /= n_total_runs
    np.fill_diagonal(co_matrix, 1.0)

    # Determine number of clusters from best kmeans run or use sqrt heuristic
    n_clusters = max(2, int(np.sqrt(n_cells / 2)))

    # Convert co_matrix to distance and apply hierarchical clustering
    dist_matrix = 1.0 - co_matrix
    dist_matrix = np.clip(dist_matrix, 0, None)
    # condense upper triangle
    from scipy.spatial.distance import squareform  # noqa: PLC0415
    dist_condensed = squareform(dist_matrix, checks=False)
    Z = linkage(dist_condensed, method="average")
    consensus_labels = fcluster(Z, n_clusters, criterion="maxclust") - 1

    return co_matrix, consensus_labels, stability_scores
