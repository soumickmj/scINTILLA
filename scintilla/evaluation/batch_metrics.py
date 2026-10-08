"""Batch correction evaluation metrics."""

from __future__ import annotations

import warnings
from typing import Optional

import numpy as np

from scintilla.config import RANDOM_SEED


def batch_asw(
    X: np.ndarray,
    batch_labels: np.ndarray,
    cell_type_labels: Optional[np.ndarray] = None,
) -> float:
    """Silhouette width of batch labels (negated so higher=better mixing).

    .. deprecated::
        Use :func:`scintilla.batch_correction.metrics.batch_asw` instead,
        which operates on AnnData and defaults to PCA space per scIB
        conventions.  This wrapper constructs a minimal AnnData and delegates
        to that implementation.

    Parameters
    ----------
    X:
        Feature matrix (cells x features).
    batch_labels:
        Batch assignment per cell.
    cell_type_labels:
        Cell type labels (unused in basic form).

    Returns
    -------
    float: negative batch ASW; higher values indicate better batch mixing.
    """
    warnings.warn(
        "scintilla.evaluation.batch_metrics.batch_asw is deprecated. "
        "Use scintilla.batch_correction.metrics.batch_asw instead, which "
        "operates on AnnData and defaults to PCA space (scIB convention).",
        DeprecationWarning,
        stacklevel=2,
    )
    import anndata as ad  # noqa: PLC0415
    from scintilla.batch_correction.metrics import batch_asw as _batch_asw  # noqa: PLC0415

    adata = ad.AnnData(X=np.asarray(X, dtype=np.float64))
    adata.obs["batch"] = np.asarray(batch_labels)
    # Use a non-existent embed_key so the corrected function falls back to adata.X
    return _batch_asw(adata, batch_key="batch", embed_key="_no_pca_")


def graph_connectivity(
    adata,
    batch_key: str,
    random_state: int = RANDOM_SEED,
) -> float:
    """Fraction of cells connected across batches in kNN graph.

    Parameters
    ----------
    adata:
        AnnData with connectivities in obsp.
    batch_key:
        Column in obs with batch labels.
    random_state:
        Random seed used when constructing a missing neighbour graph.

    Returns
    -------
    float in [0, 1].
    """
    try:
        import scanpy as sc  # noqa: PLC0415
        import scipy.sparse as sp  # noqa: PLC0415

        if "connectivities" not in adata.obsp:
            sc.pp.neighbors(adata, random_state=random_state)

        conn = adata.obsp["connectivities"]
        batch = adata.obs[batch_key].values
        unique_batches = np.unique(batch)

        if len(unique_batches) <= 1:
            return 0.0

        # Vectorised: for each cell, check if any neighbour has a different
        # batch label.  Build a sparse boolean matrix where entry (i, j) is
        # True when cell i's neighbour j belongs to the same batch.
        # Then cells with *all* neighbours in the same batch have
        # row-sum == number-of-neighbours; others are cross-connected.
        batch_codes = np.zeros(adata.n_obs, dtype=np.int32)
        for idx_b, b in enumerate(unique_batches):
            batch_codes[batch == b] = idx_b

        # For each non-zero entry (i, j) in conn, check batch_codes[i] != batch_codes[j]
        conn_coo = sp.coo_matrix(conn)
        cross_batch = batch_codes[conn_coo.row] != batch_codes[conn_coo.col]
        # Build a sparse matrix of cross-batch edges, then check which rows have any
        cross_matrix = sp.csr_matrix(
            (cross_batch.astype(np.float32), (conn_coo.row, conn_coo.col)),
            shape=conn.shape,
        )
        has_cross = np.asarray(cross_matrix.sum(axis=1)).ravel() > 0
        n_connected = int(has_cross.sum())

        return n_connected / adata.n_obs if adata.n_obs > 0 else 0.0
    except Exception:
        return float("nan")


def principal_component_regression(
    X_before: np.ndarray,
    X_after: np.ndarray,
    batch_labels: np.ndarray,
    random_state: int = RANDOM_SEED,
) -> float:
    """PCR score: reduction in variance explained by batch after correction.

    Parameters
    ----------
    X_before:
        Data before batch correction.
    X_after:
        Data after batch correction.
    batch_labels:
        Batch assignment per cell.
    random_state:
        Random seed for PCA when a randomized solver is selected.

    Returns
    -------
    float: PCR score (higher = more batch variance removed).
    """
    from sklearn.decomposition import PCA  # noqa: PLC0415
    from sklearn.linear_model import LinearRegression  # noqa: PLC0415
    from sklearn.preprocessing import LabelEncoder  # noqa: PLC0415

    le = LabelEncoder()
    batch_enc = le.fit_transform(batch_labels).reshape(-1, 1)

    def _variance_explained(X: np.ndarray) -> float:
        n_comps = min(20, X.shape[0] - 1, X.shape[1] - 1)
        if n_comps < 1:
            return float("nan")
        pca = PCA(n_components=n_comps, random_state=random_state)
        pcs = pca.fit_transform(X)
        r2_list = []
        for pc in pcs.T:
            lr = LinearRegression().fit(batch_enc, pc)
            ss_res = np.sum((pc - lr.predict(batch_enc)) ** 2)
            ss_tot = np.sum((pc - pc.mean()) ** 2)
            r2 = 1 - ss_res / (ss_tot + 1e-10)
            r2_list.append(max(r2, 0.0))
        explained = pca.explained_variance_ratio_
        return float(np.sum(np.array(r2_list) * explained))

    var_before = _variance_explained(X_before)
    var_after = _variance_explained(X_after)
    if np.isnan(var_before) or np.isnan(var_after):
        return float("nan")
    return float(var_before - var_after)
