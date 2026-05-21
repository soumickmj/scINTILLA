"""Clustering evaluation metrics."""

from __future__ import annotations

from typing import Optional

import numpy as np
from scipy.cluster.hierarchy import cophenet
from scipy.spatial.distance import pdist
from sklearn.metrics import adjusted_rand_score as _ari
from sklearn.metrics import silhouette_score as _sil

from scintilla.clustering.utils import map_clusters_to_labels


def adjusted_rand_index(true_labels: np.ndarray, pred_labels: np.ndarray) -> float:
    """Compute Adjusted Rand Index."""
    return float(_ari(true_labels, pred_labels))


def silhouette(X: np.ndarray, labels: np.ndarray) -> float:
    """Compute silhouette score (returns NaN if not computable)."""
    unique = np.unique(labels[labels != -1])
    if len(unique) < 2:
        return float("nan")
    mask = labels != -1
    return float(_sil(X[mask], labels[mask]))


def cophenetic_correlation_coefficient(
    X: np.ndarray,
    labels: Optional[np.ndarray] = None,
    Z: Optional[np.ndarray] = None,
) -> float:
    """Compute cophenetic correlation coefficient.

    Provide either a linkage matrix Z, or let the function compute one.
    """
    if Z is None:
        from scipy.cluster.hierarchy import linkage  # noqa: PLC0415
        dist = pdist(X, metric="euclidean")
        Z = linkage(dist, method="complete")
    dist = pdist(X, metric="euclidean")
    c, _ = cophenet(Z, dist)
    return float(c)


def accuracy_from_mapped_labels(
    true_labels: np.ndarray,
    pred_labels: np.ndarray,
) -> float:
    """Compute accuracy after majority-vote mapping of cluster -> true label.

    Noise points (``pred_labels == -1``, e.g. from DBSCAN) are always
    counted as incorrect predictions and are excluded from the majority-vote
    mapping so they cannot inflate accuracy.

    Note: for algorithms producing many noise points (e.g. DBSCAN with
    aggressive parameters) the denominator includes *all* cells, which may
    make accuracy appear misleadingly low.  This is intentional — noise
    points are counted as incorrect.
    """
    valid_mask = pred_labels != -1
    if not valid_mask.any():
        return 0.0
    # Build majority-vote mapping only from non-noise points
    mapping = map_clusters_to_labels(pred_labels[valid_mask], true_labels[valid_mask])
    # Count correct: noise points are never correct
    n_correct = sum(
        mapping.get(p) == t
        for p, t in zip(pred_labels, true_labels)
        if p != -1
    )
    return float(n_correct) / len(true_labels)


from sklearn.metrics import (
    normalized_mutual_info_score as _nmi,
    adjusted_mutual_info_score as _ami,
    v_measure_score as _v,
    homogeneity_score as _hom,
    completeness_score as _comp,
    fowlkes_mallows_score as _fm,
    calinski_harabasz_score as _ch,
    davies_bouldin_score as _db,
)


def normalised_mutual_info(true_labels: np.ndarray, pred_labels: np.ndarray) -> float:
    """Normalised Mutual Information."""
    return float(_nmi(true_labels, pred_labels))


def adjusted_mutual_info(true_labels: np.ndarray, pred_labels: np.ndarray) -> float:
    """Adjusted Mutual Information."""
    return float(_ami(true_labels, pred_labels))


def v_measure(true_labels: np.ndarray, pred_labels: np.ndarray) -> float:
    """V-measure score."""
    return float(_v(true_labels, pred_labels))


def homogeneity(true_labels: np.ndarray, pred_labels: np.ndarray) -> float:
    """Homogeneity score."""
    return float(_hom(true_labels, pred_labels))


def completeness(true_labels: np.ndarray, pred_labels: np.ndarray) -> float:
    """Completeness score."""
    return float(_comp(true_labels, pred_labels))


def fowlkes_mallows(true_labels: np.ndarray, pred_labels: np.ndarray) -> float:
    """Fowlkes-Mallows index."""
    return float(_fm(true_labels, pred_labels))


def calinski_harabasz(X: np.ndarray, labels: np.ndarray) -> float:
    """Calinski-Harabasz index."""
    unique = np.unique(labels[labels != -1])
    if len(unique) < 2:
        return float("nan")
    mask = labels != -1
    return float(_ch(X[mask], labels[mask]))


def davies_bouldin(X: np.ndarray, labels: np.ndarray) -> float:
    """Davies-Bouldin index."""
    unique = np.unique(labels[labels != -1])
    if len(unique) < 2:
        return float("nan")
    mask = labels != -1
    return float(_db(X[mask], labels[mask]))


def comprehensive_clustering_metrics(
    X: np.ndarray,
    true_labels: np.ndarray,
    pred_labels: np.ndarray,
    bootstrap_ci: bool = False,
    n_bootstrap: int = 2000,
    seed: int = 42,
) -> dict:
    """Compute all clustering metrics.

    Parameters
    ----------
    X:
        Feature matrix.
    true_labels:
        Ground truth labels.
    pred_labels:
        Predicted cluster labels.
    bootstrap_ci:
        If ``True``, compute 95 % bootstrap confidence intervals for
        each metric by resampling cells with their fixed labels.
    n_bootstrap:
        Number of bootstrap resamples.
    seed:
        Random seed for bootstrap.

    Returns
    -------
    dict with all metrics.  When *bootstrap_ci* is ``True``, each metric
    additionally has ``<metric>_ci_low`` and ``<metric>_ci_high`` keys.
    """
    metrics = {
        "ari": adjusted_rand_index(true_labels, pred_labels),
        "nmi": normalised_mutual_info(true_labels, pred_labels),
        "ami": adjusted_mutual_info(true_labels, pred_labels),
        "v_measure": v_measure(true_labels, pred_labels),
        "homogeneity": homogeneity(true_labels, pred_labels),
        "completeness": completeness(true_labels, pred_labels),
        "fowlkes_mallows": fowlkes_mallows(true_labels, pred_labels),
        "silhouette": silhouette(X, pred_labels),
        "calinski_harabasz": calinski_harabasz(X, pred_labels),
        "davies_bouldin": davies_bouldin(X, pred_labels),
    }

    if bootstrap_ci:
        ci_results = bootstrap_clustering_metrics(
            X, true_labels, pred_labels, B=n_bootstrap, seed=seed,
        )
        for m_name, m_ci in ci_results.items():
            metrics[f"{m_name}_ci_low"] = m_ci["ci_low"]
            metrics[f"{m_name}_ci_high"] = m_ci["ci_high"]

    return metrics


def bootstrap_clustering_metrics(
    X: np.ndarray,
    true_labels: np.ndarray,
    pred_labels: np.ndarray,
    B: int = 2000,
    alpha: float = 0.05,
    seed: int = 42,
) -> dict:
    """Bootstrap confidence intervals for clustering metrics.

    Resamples cells **with replacement** while keeping their assigned
    cluster labels and ground-truth labels fixed (the clustering is not
    re-run).  This gives a CI for the metric *conditional on the
    clustering solution*.

    Parameters
    ----------
    X:
        Feature matrix.
    true_labels:
        Ground-truth labels.
    pred_labels:
        Predicted cluster labels.
    B:
        Bootstrap replicates.
    alpha:
        Significance level.
    seed:
        Random seed.

    Returns
    -------
    dict mapping metric name → ``{ci_low, ci_high}``.
    """
    from sklearn.metrics import (  # noqa: PLC0415
        adjusted_rand_score,
        normalized_mutual_info_score,
        silhouette_score,
    )

    import logging  # noqa: PLC0415
    import warnings  # noqa: PLC0415

    logger = logging.getLogger(__name__)

    n = len(true_labels)
    rng = np.random.default_rng(seed)

    boot_ari = np.empty(B)
    boot_nmi = np.empty(B)
    boot_sil = np.empty(B)
    n_failed_ari = 0
    n_failed_nmi = 0
    n_failed_sil = 0

    for b in range(B):
        idx = rng.integers(0, n, size=n)
        t = true_labels[idx]
        p = pred_labels[idx]

        # ARI / NMI — can fail if a resample collapses a rare cluster
        try:
            boot_ari[b] = adjusted_rand_score(t, p)
        except Exception:
            boot_ari[b] = np.nan
            n_failed_ari += 1
        try:
            boot_nmi[b] = normalized_mutual_info_score(t, p)
        except Exception:
            boot_nmi[b] = np.nan
            n_failed_nmi += 1

        # Silhouette — requires ≥ 2 unique non-noise labels
        unique_p = np.unique(p[p != -1])
        if len(unique_p) >= 2:
            mask = p != -1
            try:
                boot_sil[b] = silhouette_score(X[idx][mask], p[mask])
            except Exception:
                boot_sil[b] = np.nan
                n_failed_sil += 1
        else:
            boot_sil[b] = np.nan
            n_failed_sil += 1

    # ---- One-time caveat about conditional CIs ----
    warnings.warn(
        "bootstrap_clustering_metrics: the returned confidence intervals "
        "are conditional on the provided clustering solution.  They "
        "quantify resampling variability in the *evaluation* of that "
        "clustering, NOT uncertainty in the clustering itself.",
        stacklevel=2,
    )

    # Warn if a substantial fraction of resamples failed (per-metric)
    for metric_name, n_fail in [("ARI", n_failed_ari),
                                 ("NMI", n_failed_nmi),
                                 ("silhouette", n_failed_sil)]:
        discard_rate = n_fail / B
        if discard_rate > 0.10:
            warnings.warn(
                f"bootstrap_clustering_metrics: {discard_rate:.1%} of "
                f"{metric_name} resamples failed (likely due to "
                f"rare-cluster dropout).  Consider using more bootstrap "
                f"replicates or checking for very small clusters.",
                stacklevel=2,
            )
    logger.debug(
        "bootstrap_clustering_metrics discard_rates: ARI=%.3f NMI=%.3f sil=%.3f",
        n_failed_ari / B, n_failed_nmi / B, n_failed_sil / B,
    )

    pct_lo = 100 * alpha / 2
    pct_hi = 100 * (1 - alpha / 2)

    return {
        "ari": {
            "ci_low": float(np.nanpercentile(boot_ari, pct_lo)),
            "ci_high": float(np.nanpercentile(boot_ari, pct_hi)),
        },
        "nmi": {
            "ci_low": float(np.nanpercentile(boot_nmi, pct_lo)),
            "ci_high": float(np.nanpercentile(boot_nmi, pct_hi)),
        },
        "silhouette": {
            "ci_low": float(np.nanpercentile(boot_sil, pct_lo)),
            "ci_high": float(np.nanpercentile(boot_sil, pct_hi)),
        },
    }
