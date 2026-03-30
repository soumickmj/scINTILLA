"""Label transfer between reference and query AnnData objects."""

from __future__ import annotations

import anndata as ad
import numpy as np

from scintilla.config import RANDOM_SEED


def transfer_labels(
    reference_adata: ad.AnnData,
    query_adata: ad.AnnData,
    label_col: str,
    method: str = "knn",
    n_neighbors: int = 15,
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

    Returns
    -------
    query_adata with transferred labels in obs[label_col + '_transferred'].
    """
    from sklearn.neighbors import KNeighborsClassifier  # noqa: PLC0415
    from sklearn.decomposition import PCA  # noqa: PLC0415

    query_adata = query_adata.copy()

    # Get features
    def _get_X(adata):
        if "X_pca" in adata.obsm:
            return adata.obsm["X_pca"].astype(np.float64)
        X = adata.X if not hasattr(adata.X, "toarray") else adata.X.toarray()
        return X.astype(np.float64)

    X_ref = _get_X(reference_adata)
    X_qry = _get_X(query_adata)

    # Align dimensions
    if X_ref.shape[1] != X_qry.shape[1]:
        n_c = min(X_ref.shape[1], X_qry.shape[1], 30)
        pca = PCA(n_components=n_c, random_state=RANDOM_SEED)
        X_ref = pca.fit_transform(X_ref)
        X_qry = pca.transform(X_qry) if X_qry.shape[1] >= n_c else pca.transform(
            np.pad(X_qry, ((0, 0), (0, n_c - X_qry.shape[1])))
        )

    y_ref = reference_adata.obs[label_col].values
    k = min(n_neighbors, X_ref.shape[0] - 1)
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_ref, y_ref)
    transferred = knn.predict(X_qry)
    query_adata.obs[label_col + "_transferred"] = transferred
    return query_adata
