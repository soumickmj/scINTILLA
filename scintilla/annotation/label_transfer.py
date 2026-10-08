"""Label transfer between reference and query AnnData objects."""

from __future__ import annotations

import anndata as ad
import numpy as np

from scintilla.config import RANDOM_SEED

# Ten genes is a deliberately conservative lower bound for a biologically useful
# mapping, while remaining small enough for compact synthetic regression fixtures.
_MIN_SHARED_GENES = 10


def transfer_labels(
    reference_adata: ad.AnnData,
    query_adata: ad.AnnData,
    label_col: str,
    method: str = "knn",
    n_neighbors: int = 15,
    random_state: int = RANDOM_SEED,
) -> ad.AnnData:
    """Transfer cell type labels from reference to query dataset.

    Parameters
    ----------
    reference_adata:
        Reference AnnData with labels in obs[label_col].
    query_adata:
        Query AnnData to receive transferred labels.
    label_col:
        Column in reference obs with labels.
    method:
        'knn': k-NN label transfer in PCA space.
    n_neighbors:
        Number of neighbours for kNN.
    random_state:
        Seed used by PCA when dimensionality reduction is required.

    Returns
    -------
    query_adata with transferred labels in obs[label_col + '_transferred'].
    """
    from sklearn.decomposition import PCA  # noqa: PLC0415
    from sklearn.neighbors import KNeighborsClassifier  # noqa: PLC0415

    if label_col not in reference_adata.obs.columns:
        raise ValueError(f"label_col '{label_col}' is missing from reference_adata.obs")
    if not isinstance(n_neighbors, (int, np.integer)) or n_neighbors < 1:
        raise ValueError("n_neighbors must be a positive integer")
    if reference_adata.n_obs == 0:
        raise ValueError("reference_adata must contain at least one cell")

    shared_genes = reference_adata.var_names.intersection(
        query_adata.var_names, sort=False
    )
    if len(shared_genes) == 0:
        raise ValueError("No shared genes found between reference and query AnnData objects")
    if len(shared_genes) < _MIN_SHARED_GENES:
        raise ValueError(
            f"Label transfer requires at least {_MIN_SHARED_GENES} shared genes; "
            f"found {len(shared_genes)}"
        )

    query_adata = query_adata.copy()

    def _as_dense(X):
        return X.toarray() if hasattr(X, "toarray") else X

    # Precomputed PCA coordinates are dataset-specific and therefore not comparable.
    # Subset both expression matrices in reference-gene order before fitting.
    X_ref = _as_dense(reference_adata[:, shared_genes].X).astype(np.float64)
    X_qry = _as_dense(query_adata[:, shared_genes].X).astype(np.float64)

    if X_ref.shape[1] > 30:
        n_c = min(X_ref.shape[0], X_ref.shape[1], 30)
        pca = PCA(n_components=n_c, random_state=random_state)
        X_ref = pca.fit_transform(X_ref)
        X_qry = pca.transform(X_qry)

    y_ref = reference_adata.obs[label_col].values
    k = min(n_neighbors, X_ref.shape[0])
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_ref, y_ref)
    transferred = knn.predict(X_qry)
    query_adata.obs[label_col + "_transferred"] = transferred
    return query_adata
