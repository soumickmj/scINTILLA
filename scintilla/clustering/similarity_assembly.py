"""Similarity-matrix assembly for patient-level clustering."""

from __future__ import annotations

from typing import Dict, List, Optional, Union

import anndata as ad
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.manifold import MDS

from scintilla.config import RANDOM_SEED
from scintilla.io.loaders import ensure_anndata


def build_per_celltype_clustering(
    data: Union[pd.DataFrame, ad.AnnData],
    patient_col: str,
    celltype_col: str,
    target_col: Optional[str] = None,
    method: str = "hierarchical",
    metric: str = "canberra",
    linkage: str = "complete",
    n_clusters: int = 3,
    random_state: int = RANDOM_SEED,
) -> Dict[str, np.ndarray]:
    """Cluster patients within each cell type separately.

    Returns
    -------
    dict : {cell_type: labels_array (length = n_patients)}
    """
    from scintilla.preprocessing.aggregation import aggregate_by_patient_celltype  # noqa: PLC0415

    adata = ensure_anndata(data)
    pseudo = aggregate_by_patient_celltype(adata, patient_col, celltype_col)
    patients = pseudo.index.get_level_values(patient_col).unique().tolist()
    cell_types = pseudo.index.get_level_values(celltype_col).unique().tolist()

    results: Dict[str, np.ndarray] = {}
    patient_to_idx = {pat: i for i, pat in enumerate(patients)}
    for ct in cell_types:
        ct_df = pseudo.xs(ct, level=celltype_col) if ct in pseudo.index.get_level_values(celltype_col) else None
        if ct_df is None or len(ct_df) < n_clusters:
            continue
        X = ct_df.values.astype(np.float64)
        X = np.nan_to_num(X, nan=0.0)
        nc = min(n_clusters, len(ct_df))
        if method == "hierarchical":
            from scintilla.clustering.hierarchical import hierarchical_sklearn  # noqa: PLC0415
            labels = hierarchical_sklearn(
                X, nc, metric=metric, linkage_method=linkage,
            )
        else:
            labels = KMeans(n_clusters=nc, random_state=random_state, n_init=5).fit_predict(X)

        # Align labels to the full patients list; patients absent from this
        # cell type receive the sentinel value -1.
        full_labels = np.full(len(patients), -1, dtype=int)
        for local_idx, pat in enumerate(ct_df.index.tolist()):
            if pat in patient_to_idx:
                full_labels[patient_to_idx[pat]] = labels[local_idx]
        results[ct] = full_labels
    return results


def build_similarity_matrices(
    clustering_results: Dict[str, np.ndarray],
    patients: List[str],
) -> Dict[str, np.ndarray]:
    """Build per-cell-type co-clustering similarity matrices.

    Returns
    -------
    dict : {cell_type: similarity_matrix (n_patients x n_patients)}
    """
    n = len(patients)
    sim_matrices: Dict[str, np.ndarray] = {}
    for ct, labels in clustering_results.items():
        if len(labels) != n:
            continue
        sim = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                # Sentinel -1 means the patient was absent for this cell type;
                # treat as unknown (no co-clustering evidence → similarity = 0).
                if labels[i] != -1 and labels[j] != -1 and labels[i] == labels[j]:
                    sim[i, j] = 1.0
        sim_matrices[ct] = sim
    return sim_matrices


def weighted_assembly(
    similarity_matrices: Dict[str, np.ndarray],
    weights: Optional[Dict[str, float]] = None,
) -> np.ndarray:
    """Compute a weighted average similarity matrix across cell types."""
    if not similarity_matrices:
        raise ValueError("No similarity matrices provided.")
    mats = list(similarity_matrices.items())
    n = mats[0][1].shape[0]
    combined = np.zeros((n, n))
    total_w = 0.0
    for ct, mat in mats:
        w = weights.get(ct, 1.0) if weights else 1.0
        combined += w * mat
        total_w += w
    return combined / max(total_w, 1e-8)


def patient_level_clustering(
    similarity_matrix: np.ndarray,
    n_clusters: int = 3,
    method: str = "mds_kmeans",
    random_state: int = RANDOM_SEED,
) -> np.ndarray:
    """Cluster patients using the assembled similarity matrix.

    Parameters
    ----------
    method:
        'mds_kmeans' or 'kmeans' (direct on similarity).
    """
    # Convert similarity to distance
    dist = 1.0 - similarity_matrix
    np.fill_diagonal(dist, 0.0)

    if method == "mds_kmeans":
        n_comp = min(n_clusters + 1, dist.shape[0] - 1, 10)
        mds = MDS(n_components=n_comp, dissimilarity="precomputed", random_state=random_state)
        embedding = mds.fit_transform(dist)
        labels = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10).fit_predict(embedding)
    else:
        labels = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10).fit_predict(
            similarity_matrix
        )
    return labels
