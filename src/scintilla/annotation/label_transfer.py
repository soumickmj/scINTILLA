"""Label transfer between reference and query AnnData objects."""

from __future__ import annotations

from typing import Optional

import anndata as ad
import numpy as np

from scintilla._compat import finish, get_matrix, prepare, record_params
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
    *,
    key_added: Optional[str] = None,
    copy: bool = False,
) -> Optional[ad.AnnData]:
    """Transfer cell type labels from a reference to a query dataset.

    Genes are matched by name: both objects are restricted to the genes they share
    (in the reference's order) before anything is fitted, and precomputed PCA
    coordinates are ignored because they are dataset specific.

    Parameters
    ----------
    reference_adata
        Reference AnnData with labels in ``obs[label_col]``.  Not modified.
    query_adata
        Query AnnData to receive the transferred labels.
    label_col
        Column in the reference ``obs`` with the labels.
    method
        ``"knn"``: k-nearest-neighbour label transfer, in PCA space when more than 30
        genes are shared.
    n_neighbors
        Number of neighbours for kNN.
    random_state
        Seed used by PCA when dimensionality reduction is required.
    key_added
        Name of the ``query_adata.obs`` column that receives the labels; defaults to
        ``label_col + "_transferred"``.
    copy
        Return a modified copy of the query instead of modifying it in place.

    Returns
    -------
    anndata.AnnData or None
        ``None`` when working in place; the annotated copy of the query when ``copy=True``.

    Raises
    ------
    ValueError
        If the objects share no genes, or fewer than 10.
    """
    from sklearn.decomposition import PCA
    from sklearn.neighbors import KNeighborsClassifier

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

    query_adata, give_back = prepare(query_adata, copy=copy)

    # Precomputed PCA coordinates are dataset-specific and therefore not comparable.
    # Subset both expression matrices in reference-gene order before fitting.
    X_ref = get_matrix(reference_adata[:, shared_genes], reason="PCA/kNN transfer needs a dense matrix", dtype=np.float64)
    X_qry = get_matrix(query_adata[:, shared_genes], reason="PCA/kNN transfer needs a dense matrix", dtype=np.float64)

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
    key_added = key_added or label_col + "_transferred"
    query_adata.obs[key_added] = transferred
    record_params(
        query_adata, key_added, label_col=label_col, method=method, n_neighbors=n_neighbors,
        n_shared_genes=len(shared_genes), random_state=random_state, key_added=key_added,
    )
    return finish(query_adata, give_back)
