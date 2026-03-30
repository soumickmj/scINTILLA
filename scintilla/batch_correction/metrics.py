"""Batch correction evaluation metrics."""

from __future__ import annotations

from typing import Optional

import numpy as np


def batch_asw(
    adata,
    batch_key: str,
    label_key: Optional[str] = None,
    embed_key: str = "X_pca",
) -> float:
    """Silhouette width of batch labels (negated so higher = better mixing).

    The raw silhouette score on batch labels ranges from -1 to 1, where
    high values indicate cells cluster tightly by batch (poor mixing).
    This function **negates** the score so that:

    * Values near **0** indicate batches are well-mixed.
    * Values near **-1** indicate cells cluster strongly by batch.

    This convention follows scIB (Luecken et al., 2022).

    By default computes ASW on a PCA embedding (``adata.obsm[embed_key]``)
    rather than the full gene-expression matrix, following scIB conventions.
    Falls back to ``adata.X`` when the embedding is not available.

    Parameters
    ----------
    adata:
        AnnData.
    batch_key:
        Column in obs with batch labels.
    label_key:
        Unused; included for API consistency.
    embed_key:
        Key in ``adata.obsm`` for the embedding to use (default ``"X_pca"``).

    Returns
    -------
    float: negative batch ASW (higher = better mixing).
    """
    from sklearn.metrics import silhouette_score  # noqa: PLC0415

    if embed_key in getattr(adata, "obsm", {}):
        X = np.asarray(adata.obsm[embed_key], dtype=np.float64)
    else:
        X = adata.X if not hasattr(adata.X, "toarray") else adata.X.toarray()
        X = X.astype(np.float64)
    batch = adata.obs[batch_key].values
    unique = np.unique(batch)
    if len(unique) < 2:
        return float("nan")
    try:
        return -float(silhouette_score(X, batch))
    except Exception:
        return float("nan")


def lisi_score(
    adata,
    batch_key: str,
    label_key: Optional[str] = None,
    n_neighbors: int = 30,
    embed_key: str = "X_pca",
) -> float:
    """Approximate LISI score using kNN locality.

    Higher values indicate better batch mixing.

    This is a Simpson-index approximation of the Local Inverse Simpson
    Index (Korsunsky et al., 2019).  Scores are not directly comparable
    to the perplexity-based LISI values from the original paper but are
    well-correlated and much faster to compute.

    By default the kNN graph is built on a PCA embedding
    (``adata.obsm[embed_key]``), falling back to ``adata.X`` when
    unavailable.

    Parameters
    ----------
    adata:
        AnnData.
    batch_key:
        Column in obs with batch labels.
    label_key:
        Unused; included for API consistency.
    n_neighbors:
        Number of neighbours.
    embed_key:
        Key in ``adata.obsm`` for the embedding to use (default ``"X_pca"``).

    Returns
    -------
    float in [1, n_batches].
    """
    from sklearn.neighbors import NearestNeighbors  # noqa: PLC0415
    import pandas as pd  # noqa: PLC0415

    if embed_key in getattr(adata, "obsm", {}):
        X = np.asarray(adata.obsm[embed_key], dtype=np.float64)
    else:
        X = adata.X if not hasattr(adata.X, "toarray") else adata.X.toarray()
        X = X.astype(np.float64)
    batch = adata.obs[batch_key].values
    unique_batches = np.unique(batch)
    n_batches = len(unique_batches)

    if n_batches < 2:
        return 1.0

    k = min(n_neighbors, X.shape[0] - 1)
    nn = NearestNeighbors(n_neighbors=k).fit(X)
    _, indices = nn.kneighbors(X)

    lisi_values = []
    for i in range(X.shape[0]):
        neighbors_batch = batch[indices[i]]
        # Simpson index
        counts = pd.Series(neighbors_batch).value_counts(normalize=True)
        simpson = (counts ** 2).sum()
        lisi_values.append(1.0 / (simpson + 1e-10))

    return float(np.mean(lisi_values))


def kbet_score(
    X: np.ndarray,
    batch_labels: np.ndarray,
    k: int = 20,
) -> float:
    """Approximate kBET score using chi-squared test on local batch composition.

    Parameters
    ----------
    X:
        Feature matrix.
    batch_labels:
        Batch assignment per cell.
    k:
        Number of neighbours.

    Returns
    -------
    float: acceptance rate in [0, 1] (higher = better mixing).
    """
    from sklearn.neighbors import NearestNeighbors  # noqa: PLC0415
    from scipy.stats import chi2  # noqa: PLC0415
    import pandas as pd  # noqa: PLC0415

    unique_batches, counts = np.unique(batch_labels, return_counts=True)
    expected_frac = counts / counts.sum()
    n_batches = len(unique_batches)

    if n_batches < 2:
        return 1.0

    k = min(k, X.shape[0] - 1)
    nn = NearestNeighbors(n_neighbors=k).fit(X)
    _, indices = nn.kneighbors(X)

    n_accepted = 0
    alpha = 0.05
    for i in range(X.shape[0]):
        nb = batch_labels[indices[i]]
        observed = np.array([np.sum(nb == b) for b in unique_batches])
        expected = expected_frac * k
        # chi-squared statistic
        chi2_stat = np.sum((observed - expected) ** 2 / (expected + 1e-10))
        p_value = 1 - chi2.cdf(chi2_stat, df=n_batches - 1)
        if p_value >= alpha:
            n_accepted += 1

    return n_accepted / X.shape[0]


def bio_conservation_score(
    adata,
    label_key: str,
    embed_key: str = "X_pca",
) -> float:
    """Cell-type silhouette score measuring biological conservation.

    Higher values indicate that cell-type structure is well-preserved
    after batch correction.

    Parameters
    ----------
    adata:
        AnnData (batch-corrected).
    label_key:
        Column in obs with cell-type labels.
    embed_key:
        Key in ``adata.obsm`` for the embedding.

    Returns
    -------
    float: silhouette score on cell-type labels (higher = better conservation).
    """
    from sklearn.metrics import silhouette_score  # noqa: PLC0415

    if embed_key in getattr(adata, "obsm", {}):
        X = np.asarray(adata.obsm[embed_key], dtype=np.float64)
    else:
        X = adata.X if not hasattr(adata.X, "toarray") else adata.X.toarray()
        X = X.astype(np.float64)

    labels = adata.obs[label_key].values
    unique = np.unique(labels)
    if len(unique) < 2:
        return float("nan")
    try:
        return float(silhouette_score(X, labels))
    except Exception:
        return float("nan")
