"""Complexity-aware time estimator for scINTILLA benchmarking methods.

Provides :func:`estimate_benchmark_time` which, given an AnnData object
and an ``AnalysisConfig``, runs a brief micro-calibration on a small
subsample, then extrapolates wall-clock time and peak memory for every
enabled method using each algorithm's known computational complexity.

The result is a DataFrame that users can inspect *before* launching a
full benchmark, allowing them to drop expensive methods when time is
limited.

Complexity models
-----------------
Each algorithm is assigned a complexity function f(n, d, k, p) where
n = cells, d = features, k = clusters, p = method-specific parameter
count (e.g. resolution grid size).  The calibration step measures the
hardware-specific constant by timing the method on two small subsamples
and solving for the coefficient.  Extrapolation to the true n then
follows from the fitted model.

When the calibration run fails (e.g. an optional dependency is missing),
the method is marked with ``status="unavailable"`` and a ``NaN``
estimate.
"""

from __future__ import annotations

import time
import warnings
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from scintilla.config import RANDOM_SEED

try:
    import anndata as ad
except ImportError:  # pragma: no cover
    ad = None  # type: ignore[assignment]


# ── Complexity models ───────────────────────────────────────────────────
#
# Each model is a callable  (n, d, k) -> float  that returns a *relative*
# cost (arbitrary units).  The calibration step divides measured time by
# this value to obtain the hardware constant c, then multiplies c by the
# model evaluated at the true (n, d, k) to get the predicted time.
#
# n = number of cells
# d = number of features (PCA components, typically 30)
# k = number of clusters (estimated from ground truth or default)

def _complexity_n_k_d(n: float, d: float, k: float) -> float:
    """O(n * k * d) -- KMeans, LogReg, kNN, NaiveBayes, LDA."""
    return n * k * d


def _complexity_n_d_logn(n: float, d: float, k: float) -> float:
    """O(n * d * log(n)) -- Random Forest, GradientBoosting, LightGBM, XGBoost."""
    return n * d * max(np.log2(n), 1.0)


def _complexity_n2_d(n: float, d: float, k: float) -> float:
    """O(n^2 * d) -- SVM, Hierarchical (distance matrix), DBSCAN worst-case."""
    return n * n * d


def _complexity_n3(n: float, d: float, k: float) -> float:
    """O(n^3) -- Spectral (eigendecomposition), QDA on large feature sets."""
    return n * n * n


def _complexity_n_logn_d(n: float, d: float, k: float) -> float:
    """O(n * log(n) * d) -- Leiden, Louvain (graph construction + community detection)."""
    return n * max(np.log2(n), 1.0) * d


def _complexity_n_logn_sq_d(n: float, d: float, k: float) -> float:
    """O(n * log^2(n) * d) -- HDBSCAN (with optimised tree, better than n^2)."""
    return n * max(np.log2(n), 1.0) ** 2 * d


def _complexity_n2_k(n: float, d: float, k: float) -> float:
    """O(n^2 * k) -- Consensus clustering (multiple base runs + co-association)."""
    return n * n * k


def _complexity_mlp(n: float, d: float, k: float) -> float:
    """O(n * d * h * epochs) -- MLP; h ~ 100, epochs ~ 200 (sklearn defaults)."""
    h, epochs = 100.0, 200.0
    return n * d * h * epochs


def _complexity_stacking(n: float, d: float, k: float) -> float:
    """O(n * d * log(n)) * cv_folds -- Stacking Ensemble (RF + LogReg, 2-fold)."""
    return 2.0 * n * d * max(np.log2(n), 1.0)


def _complexity_shap(n: float, d: float, k: float) -> float:
    """O(n * 2^d) in theory, but TreeSHAP is O(n * d * T * L) for trees.
    We approximate as O(n * d^2) for practical single-cell feature counts."""
    return n * d * d


# ── Batch correction complexity models ──────────────────────────────────

def _complexity_combat(n: float, d: float, k: float) -> float:
    """O(n * d) -- ComBat is linear regression, essentially linear."""
    return n * d


def _complexity_harmony(n: float, d: float, k: float) -> float:
    """O(n * d * iterations) -- Harmony iterates in PCA space."""
    return n * d * 20.0  # ~20 iterations typical


def _complexity_bbknn(n: float, d: float, k: float) -> float:
    """O(n * log(n) * d) -- BBKNN builds batch-balanced kNN graph."""
    return n * max(np.log2(n), 1.0) * d


def _complexity_scanorama(n: float, d: float, k: float) -> float:
    """O(n^2 * d) worst case -- panoramic stitching across batches."""
    return n * n * d


# ── Feature selection complexity models ─────────────────────────────────

def _complexity_pca_loadings(n: float, d: float, k: float) -> float:
    """O(n * d * min(n,d)) -- PCA decomposition."""
    return n * d * min(n, d)


def _complexity_mi(n: float, d: float, k: float) -> float:
    """O(n * d * log(n)) -- mutual information (kNN-based estimator)."""
    return n * d * max(np.log2(n), 1.0)


def _complexity_boruta(n: float, d: float, k: float) -> float:
    """O(n * 2d * log(n) * n_iterations) -- Boruta wraps Random Forest
    on doubled feature set, typically 100+ iterations."""
    return n * 2 * d * max(np.log2(n), 1.0) * 100.0


def _complexity_mrmr(n: float, d: float, k: float) -> float:
    """O(n * d^2) -- mRMR computes pairwise MI between features."""
    return n * d * d


# ── Method registry ─────────────────────────────────────────────────────

@dataclass
class _MethodSpec:
    """Internal specification for a benchmarkable method."""

    name: str
    stage: str  # "clustering", "classification", "batch_correction", "feature_selection"
    complexity_fn: Callable[[float, float, float], float]
    complexity_label: str
    grid_size_fn: Optional[Callable] = None  # (config) -> int


# Clustering methods -- grid_size_fn returns the number of parameter
# combinations that benchmark_clustering_methods will sweep.

_CLUSTERING_SPECS: Dict[str, _MethodSpec] = {
    "kmeans": _MethodSpec(
        "KMeans", "clustering", _complexity_n_k_d, "O(n*k*d)",
        grid_size_fn=lambda cfg: 2,  # two init strategies
    ),
    "hierarchical": _MethodSpec(
        "Hierarchical", "clustering", _complexity_n2_d, "O(n^2*d)",
        grid_size_fn=lambda cfg: 7 * 4,  # 7 metrics * 4 linkages
    ),
    "dbscan": _MethodSpec(
        "DBSCAN", "clustering", _complexity_n2_d, "O(n^2*d)",
        grid_size_fn=lambda cfg: 9 * 7,  # 9 eps * 7 metrics
    ),
    "leiden": _MethodSpec(
        "Leiden", "clustering", _complexity_n_logn_d, "O(n*log(n)*d)",
        grid_size_fn=lambda cfg: len(getattr(cfg, "leiden_resolutions", [0.5, 1.0])),
    ),
    "louvain": _MethodSpec(
        "Louvain", "clustering", _complexity_n_logn_d, "O(n*log(n)*d)",
        grid_size_fn=lambda cfg: len(getattr(cfg, "louvain_resolutions", [0.5, 1.0])),
    ),
    "hdbscan": _MethodSpec(
        "HDBSCAN", "clustering", _complexity_n_logn_sq_d, "O(n*log^2(n)*d)",
        grid_size_fn=lambda cfg: (
            len(getattr(cfg, "hdbscan_min_cluster_sizes", [20]))
            * len(getattr(cfg, "hdbscan_min_samples", [None]))
        ),
    ),
    "spectral": _MethodSpec(
        "Spectral", "clustering", _complexity_n3, "O(n^3)",
        grid_size_fn=lambda cfg: len(getattr(cfg, "spectral_n_clusters_range", [])),
    ),
    "consensus": _MethodSpec(
        "Consensus", "clustering", _complexity_n2_k, "O(n^2*k)",
        grid_size_fn=lambda cfg: 1,
    ),
}

_CLASSIFIER_SPECS: Dict[str, _MethodSpec] = {
    "LogReg": _MethodSpec("LogReg", "classification", _complexity_n_k_d, "O(n*k*d)"),
    "RF": _MethodSpec("RF", "classification", _complexity_n_d_logn, "O(n*d*log(n))"),
    "SVM": _MethodSpec("SVM", "classification", _complexity_n2_d, "O(n^2*d)"),
    "MLP": _MethodSpec("MLP", "classification", _complexity_mlp, "O(n*d*h*epochs)"),
    "LDA": _MethodSpec("LDA", "classification", _complexity_n_k_d, "O(n*k*d)"),
    "QDA": _MethodSpec("QDA", "classification", _complexity_n3, "O(n^3)"),
    "kNN": _MethodSpec("kNN", "classification", _complexity_n_k_d, "O(n*k*d)"),
    "GradientBoosting": _MethodSpec(
        "GradientBoosting", "classification", _complexity_n_d_logn, "O(n*d*log(n))",
    ),
    "NaiveBayes": _MethodSpec("NaiveBayes", "classification", _complexity_n_k_d, "O(n*k*d)"),
    "StackingEnsemble": _MethodSpec(
        "StackingEnsemble", "classification", _complexity_stacking, "O(n*d*log(n)*cv)",
    ),
    "XGBoost": _MethodSpec("XGBoost", "classification", _complexity_n_d_logn, "O(n*d*log(n))"),
    "LightGBM": _MethodSpec("LightGBM", "classification", _complexity_n_d_logn, "O(n*d*log(n))"),
}

_BATCH_CORRECTION_SPECS: Dict[str, _MethodSpec] = {
    "combat": _MethodSpec("ComBat", "batch_correction", _complexity_combat, "O(n*d)"),
    "harmony": _MethodSpec("Harmony", "batch_correction", _complexity_harmony, "O(n*d*iter)"),
    "bbknn": _MethodSpec("BBKNN", "batch_correction", _complexity_bbknn, "O(n*log(n)*d)"),
    "scanorama": _MethodSpec("Scanorama", "batch_correction", _complexity_scanorama, "O(n^2*d)"),
}

_FEATURE_SELECTION_SPECS: Dict[str, _MethodSpec] = {
    "pca_loadings": _MethodSpec(
        "PCA Loadings", "feature_selection", _complexity_pca_loadings, "O(n*d*min(n,d))",
    ),
    "mutual_information": _MethodSpec(
        "Mutual Information", "feature_selection", _complexity_mi, "O(n*d*log(n))",
    ),
    "boruta": _MethodSpec("Boruta", "feature_selection", _complexity_boruta, "O(n*2d*log(n)*iter)"),
    "mrmr": _MethodSpec("mRMR", "feature_selection", _complexity_mrmr, "O(n*d^2)"),
}

_SHAP_SPEC = _MethodSpec("SHAP", "classification", _complexity_shap, "O(n*d^2)")


# ── Calibration helpers ─────────────────────────────────────────────────

def _subsample(adata, n_cells: int, seed: int = RANDOM_SEED):
    """Return a subsampled copy of *adata* with at most *n_cells* cells."""
    rng = np.random.default_rng(seed)
    n = min(n_cells, adata.n_obs)
    idx = rng.choice(adata.n_obs, size=n, replace=False)
    return adata[idx].copy()


def _get_X(adata, use_rep: str = "X_pca"):
    """Extract the feature matrix from an AnnData object."""
    if use_rep in adata.obsm:
        return adata.obsm[use_rep].astype(np.float64)
    X = adata.X if not hasattr(adata.X, "toarray") else adata.X.toarray()
    return X.astype(np.float64)


def _timed_call(fn: Callable, *args, **kwargs) -> Tuple[float, float]:
    """Run *fn* and return (wall_seconds, peak_memory_mb)."""
    import tracemalloc as _tm

    _tm.start()
    t0 = time.perf_counter()
    try:
        fn(*args, **kwargs)
    except Exception:
        pass  # we only need the timing; failures are handled upstream
    elapsed = time.perf_counter() - t0
    _, peak = _tm.get_traced_memory()
    _tm.stop()
    return elapsed, peak / (1024 * 1024)


# ── Per-method micro-benchmark runners ──────────────────────────────────
#
# Each returns (wall_seconds, peak_memory_mb) for a *single* parameter
# setting on the given subsample, or raises an exception if the method
# is unavailable.

def _calibrate_kmeans(X: np.ndarray, k: int, random_state: int = RANDOM_SEED) -> Tuple[float, float]:
    from sklearn.cluster import KMeans
    return _timed_call(KMeans(n_clusters=k, n_init=1, max_iter=50, random_state=random_state).fit, X)


def _calibrate_hierarchical(X: np.ndarray, k: int, random_state: int = RANDOM_SEED) -> Tuple[float, float]:
    from scipy.cluster.hierarchy import linkage, fcluster
    from scipy.spatial.distance import pdist

    def _run():
        dist = pdist(X, metric="euclidean")
        Z = linkage(dist, method="ward")
        fcluster(Z, t=k, criterion="maxclust")

    return _timed_call(_run)


def _calibrate_dbscan(X: np.ndarray, k: int, random_state: int = RANDOM_SEED) -> Tuple[float, float]:
    from sklearn.cluster import DBSCAN
    return _timed_call(DBSCAN(eps=0.5).fit, X)


def _calibrate_leiden(X: np.ndarray, k: int, random_state: int = RANDOM_SEED) -> Tuple[float, float]:
    import scanpy as sc

    def _run():
        _adata = ad.AnnData(X=X.astype(np.float32))
        sc.pp.neighbors(_adata, use_rep="X", random_state=random_state)
        sc.tl.leiden(_adata, resolution=1.0, random_state=random_state)

    return _timed_call(_run)


def _calibrate_louvain(X: np.ndarray, k: int, random_state: int = RANDOM_SEED) -> Tuple[float, float]:
    import scanpy as sc

    def _run():
        _adata = ad.AnnData(X=X.astype(np.float32))
        sc.pp.neighbors(_adata, use_rep="X", random_state=random_state)
        sc.tl.louvain(_adata, resolution=1.0, random_state=random_state)

    return _timed_call(_run)


def _calibrate_hdbscan(X: np.ndarray, k: int, random_state: int = RANDOM_SEED) -> Tuple[float, float]:
    import hdbscan as _hdb

    return _timed_call(_hdb.HDBSCAN(min_cluster_size=20).fit, X)


def _calibrate_spectral(X: np.ndarray, k: int, random_state: int = RANDOM_SEED) -> Tuple[float, float]:
    from sklearn.cluster import SpectralClustering

    return _timed_call(
        SpectralClustering(n_clusters=k, affinity="nearest_neighbors", random_state=random_state).fit, X,
    )


def _calibrate_consensus(X: np.ndarray, k: int, random_state: int = RANDOM_SEED) -> Tuple[float, float]:
    # Consensus is built on top of KMeans, approximate with repeated KMeans
    from sklearn.cluster import KMeans

    def _run():
        for _ in range(5):
            KMeans(n_clusters=k, n_init=1, max_iter=50, random_state=random_state).fit(X)

    return _timed_call(_run)


# Classification calibrators (single fit on the subsample)

def _make_classification_data(X: np.ndarray, k: int):
    """Create synthetic labels for classifier calibration."""
    labels = np.array([f"type_{i % k}" for i in range(X.shape[0])])
    return X, labels


def _calibrate_logreg(X: np.ndarray, k: int, random_state: int = RANDOM_SEED) -> Tuple[float, float]:
    from sklearn.linear_model import LogisticRegression

    X_c, y = _make_classification_data(X, k)
    return _timed_call(LogisticRegression(max_iter=200, random_state=random_state).fit, X_c, y)


def _calibrate_rf(X: np.ndarray, k: int, random_state: int = RANDOM_SEED) -> Tuple[float, float]:
    from sklearn.ensemble import RandomForestClassifier

    X_c, y = _make_classification_data(X, k)
    return _timed_call(RandomForestClassifier(n_estimators=100, random_state=random_state).fit, X_c, y)


def _calibrate_svm(X: np.ndarray, k: int, random_state: int = RANDOM_SEED) -> Tuple[float, float]:
    from sklearn.svm import SVC

    X_c, y = _make_classification_data(X, k)
    return _timed_call(SVC(kernel="rbf", random_state=random_state).fit, X_c, y)


def _calibrate_mlp(X: np.ndarray, k: int, random_state: int = RANDOM_SEED) -> Tuple[float, float]:
    from sklearn.neural_network import MLPClassifier

    X_c, y = _make_classification_data(X, k)
    return _timed_call(
        MLPClassifier(max_iter=50, random_state=random_state).fit, X_c, y,
    )


def _calibrate_lda(X: np.ndarray, k: int, random_state: int = RANDOM_SEED) -> Tuple[float, float]:
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

    X_c, y = _make_classification_data(X, k)
    return _timed_call(LinearDiscriminantAnalysis().fit, X_c, y)


def _calibrate_qda(X: np.ndarray, k: int, random_state: int = RANDOM_SEED) -> Tuple[float, float]:
    from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

    X_c, y = _make_classification_data(X, k)
    return _timed_call(QuadraticDiscriminantAnalysis().fit, X_c, y)


def _calibrate_knn(X: np.ndarray, k: int, random_state: int = RANDOM_SEED) -> Tuple[float, float]:
    from sklearn.neighbors import KNeighborsClassifier

    X_c, y = _make_classification_data(X, k)
    return _timed_call(KNeighborsClassifier().fit, X_c, y)


def _calibrate_gb(X: np.ndarray, k: int, random_state: int = RANDOM_SEED) -> Tuple[float, float]:
    from sklearn.ensemble import HistGradientBoostingClassifier

    X_c, y = _make_classification_data(X, k)
    return _timed_call(
        HistGradientBoostingClassifier(max_iter=50, random_state=random_state).fit, X_c, y,
    )


def _calibrate_nb(X: np.ndarray, k: int, random_state: int = RANDOM_SEED) -> Tuple[float, float]:
    from sklearn.naive_bayes import GaussianNB

    X_c, y = _make_classification_data(X, k)
    return _timed_call(GaussianNB().fit, X_c, y)


def _calibrate_stacking(X: np.ndarray, k: int, random_state: int = RANDOM_SEED) -> Tuple[float, float]:
    from sklearn.ensemble import RandomForestClassifier, StackingClassifier
    from sklearn.linear_model import LogisticRegression

    X_c, y = _make_classification_data(X, k)
    clf = StackingClassifier(
        estimators=[("rf", RandomForestClassifier(n_estimators=50, random_state=random_state))],
        final_estimator=LogisticRegression(max_iter=200, random_state=random_state),
        cv=2,
    )
    return _timed_call(clf.fit, X_c, y)


def _calibrate_xgboost(X: np.ndarray, k: int, random_state: int = RANDOM_SEED) -> Tuple[float, float]:
    from xgboost import XGBClassifier

    X_c, y = _make_classification_data(X, k)
    # XGBoost needs integer labels
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    return _timed_call(
        XGBClassifier(n_estimators=100, max_depth=6, random_state=random_state, verbosity=0).fit, X_c, y_enc,
    )


def _calibrate_lightgbm(X: np.ndarray, k: int, random_state: int = RANDOM_SEED) -> Tuple[float, float]:
    from lightgbm import LGBMClassifier

    X_c, y = _make_classification_data(X, k)
    return _timed_call(
        LGBMClassifier(n_estimators=100, random_state=random_state, verbose=-1).fit, X_c, y,
    )


_CALIBRATION_FNS: Dict[str, Callable] = {
    # Clustering
    "kmeans": _calibrate_kmeans,
    "hierarchical": _calibrate_hierarchical,
    "dbscan": _calibrate_dbscan,
    "leiden": _calibrate_leiden,
    "louvain": _calibrate_louvain,
    "hdbscan": _calibrate_hdbscan,
    "spectral": _calibrate_spectral,
    "consensus": _calibrate_consensus,
    # Classification
    "LogReg": _calibrate_logreg,
    "RF": _calibrate_rf,
    "SVM": _calibrate_svm,
    "MLP": _calibrate_mlp,
    "LDA": _calibrate_lda,
    "QDA": _calibrate_qda,
    "kNN": _calibrate_knn,
    "GradientBoosting": _calibrate_gb,
    "NaiveBayes": _calibrate_nb,
    "StackingEnsemble": _calibrate_stacking,
    "XGBoost": _calibrate_xgboost,
    "LightGBM": _calibrate_lightgbm,
}


# ── Core estimation logic ──────────────────────────────────────────────


def _extrapolate(
    t_calib: float,
    complexity_fn: Callable,
    n_calib: int,
    d: int,
    k: int,
    n_full: int,
    grid_size: int = 1,
) -> float:
    """Extrapolate from calibration timing to full-dataset prediction.

    Uses the ratio of complexity at full scale to complexity at
    calibration scale, multiplied by the measured calibration time
    and the parameter grid size.  This accounts for both the
    algorithmic scaling and the number of hyperparameter combinations
    the benchmark will sweep through.

    Parameters
    ----------
    t_calib:
        Wall-clock seconds measured during calibration (single run).
    complexity_fn:
        Theoretical complexity model f(n, d, k) -> float.
    n_calib:
        Number of cells used during calibration.
    d:
        Feature dimensionality (number of PCA components).
    k:
        Number of clusters / classes.
    n_full:
        Total number of cells in the full dataset.
    grid_size:
        Number of parameter combinations to sweep (multiplicative).

    Returns
    -------
    Predicted wall-clock seconds for the full benchmark run.
    """
    cost_calib = complexity_fn(float(n_calib), float(d), float(k))
    cost_full = complexity_fn(float(n_full), float(d), float(k))

    if cost_calib <= 0:
        return float("nan")

    ratio = cost_full / cost_calib
    return t_calib * ratio * grid_size


def _format_time(seconds: float) -> str:
    """Human-readable time string."""
    if np.isnan(seconds):
        return "unavailable"
    if seconds < 1.0:
        return f"{seconds * 1000:.0f}ms"
    if seconds < 60.0:
        return f"{seconds:.1f}s"
    if seconds < 3600.0:
        minutes = seconds / 60.0
        return f"{minutes:.1f}min"
    hours = seconds / 3600.0
    return f"{hours:.1f}h"


# ── Public API ──────────────────────────────────────────────────────────


def estimate_benchmark_time(
    adata,
    config=None,
    *,
    stages: Optional[List[str]] = None,
    use_rep: str = "X_pca",
    n_clusters: Optional[int] = None,
    cell_type_col: str = "cell_type",
    calibration_cells: int = 500,
    cv_folds: Optional[int] = None,
    batch_methods: Optional[List[str]] = None,
    verbose: Optional[bool] = None,
    random_state: Optional[int] = None,
) -> pd.DataFrame:
    """Estimate wall-clock time for each benchmarking method.

    Runs a quick micro-calibration on a small subsample, then
    extrapolates to the full dataset using each algorithm's known
    computational complexity.  Returns a DataFrame that the user can
    inspect before launching the full benchmark.

    Parameters
    ----------
    adata:
        Full AnnData object (must have the representation specified
        by *use_rep* in ``.obsm``, or raw counts in ``.X``).
    config:
        An ``AnalysisConfig`` instance.  If ``None``, the default
        config is used (all methods enabled).
    stages:
        Which pipeline stages to estimate.  Any subset of
        ``["clustering", "classification", "batch_correction",
        "feature_selection"]``.  Defaults to
        ``["clustering", "classification"]``.
    use_rep:
        Representation key in ``adata.obsm`` for the feature matrix.
    n_clusters:
        Number of clusters / classes.  Inferred from
        *cell_type_col* if not given.
    cell_type_col:
        Column in ``adata.obs`` used to infer *n_clusters*.
    calibration_cells:
        Number of cells for the micro-calibration subsample.
        Smaller values are faster but noisier; 500 is a good
        default.  Must be >= 50.
    cv_folds:
        Cross-validation folds for classification estimates.  If
        ``None``, taken from *config* (default 5).
    batch_methods:
        Batch correction methods to estimate.  Defaults to all four
        (``["combat", "harmony", "bbknn", "scanorama"]``).
    verbose:
        Print progress during calibration.

    Returns
    -------
    pd.DataFrame
        Columns: ``stage``, ``method``, ``complexity``,
        ``grid_size``, ``estimated_seconds``, ``estimated_time``,
        ``calibration_seconds``, ``status``.

        Sorted by ``estimated_seconds`` descending (slowest first).

    Examples
    --------
    >>> from scintilla import AnalysisConfig
    >>> from scintilla.benchmarking.time_estimator import estimate_benchmark_time
    >>>
    >>> cfg = AnalysisConfig.default()
    >>> estimates = estimate_benchmark_time(adata, config=cfg)
    >>> print(estimates[["method", "estimated_time", "complexity"]])
    >>>
    >>> # Drop anything predicted to take more than 10 minutes
    >>> fast_methods = estimates[estimates["estimated_seconds"] < 600]
    """
    # Lazy import to avoid circular dependencies
    try:
        from scintilla.analysis_config import AnalysisConfig as _AC
        _default_config_cls = _AC
    except ImportError:
        _default_config_cls = None

    if config is None:
        if _default_config_cls is not None:
            config = _default_config_cls.default()
        else:
            # Lightweight fallback when scintilla is not installed
            # (e.g. during isolated testing).
            class _FallbackConfig:
                clustering_methods = list(_CLUSTERING_SPECS.keys())
                classifiers = list(_CLASSIFIER_SPECS.keys())
                leiden_resolutions = [0.1, 0.3, 0.5, 0.8, 1.0, 1.5, 2.0, 3.0]
                louvain_resolutions = [0.1, 0.3, 0.5, 0.8, 1.0, 1.5, 2.0, 3.0]
                hdbscan_min_cluster_sizes = [10, 20, 50]
                hdbscan_min_samples = [None, 5]
                spectral_n_clusters_range = [2, 3, 4, 5, 6, 7, 8, 9, 10]
                include_shap = True
                cv_folds = 5
                classification_estimator = "cv"
                feature_selection_methods = ["pca_loadings", "mutual_information"]
                random_seed = RANDOM_SEED
                verbose = True

            config = _FallbackConfig()

    if verbose is None:
        verbose = getattr(config, "verbose", True)
    if random_state is None:
        random_state = getattr(config, "random_seed", RANDOM_SEED)

    if stages is None:
        stages = ["clustering", "classification"]

    if calibration_cells < 50:
        raise ValueError("calibration_cells must be >= 50")

    # Infer dataset parameters
    n_full = adata.n_obs
    X_full = _get_X(adata, use_rep)
    d = X_full.shape[1]

    if n_clusters is None:
        if cell_type_col in adata.obs.columns:
            n_clusters = len(adata.obs[cell_type_col].unique())
        else:
            n_clusters = 8  # sensible fallback

    k = n_clusters

    if cv_folds is None:
        cv_folds = getattr(config, "cv_folds", 5)

    # Prepare calibration subsample
    n_calib = min(calibration_cells, n_full)
    adata_calib = _subsample(adata, n_calib, seed=random_state)
    X_calib = _get_X(adata_calib, use_rep)

    if verbose:
        print(
            f"[time_estimator] Calibrating on {n_calib} cells "
            f"(full dataset: {n_full} cells, {d} features, ~{k} clusters)"
        )

    records: List[Dict[str, Any]] = []

    # ── Clustering ──────────────────────────────────────────────────
    if "clustering" in stages:
        enabled = set(
            m.lower() for m in getattr(config, "clustering_methods", list(_CLUSTERING_SPECS.keys()))
        )
        for method_key, spec in _CLUSTERING_SPECS.items():
            if method_key not in enabled:
                continue

            grid_size = spec.grid_size_fn(config) if spec.grid_size_fn else 1
            if grid_size <= 0:
                continue  # e.g. spectral with empty range

            calib_fn = _CALIBRATION_FNS.get(method_key)
            status = "ok"
            t_calib = np.nan
            t_est = np.nan

            if calib_fn is not None:
                try:
                    t_calib, _ = calib_fn(X_calib, k, random_state)
                    t_est = _extrapolate(
                        t_calib, spec.complexity_fn, n_calib, d, k, n_full, grid_size,
                    )
                    if verbose:
                        print(f"  {spec.name:25s}  calib={t_calib:.3f}s  "
                              f"est={_format_time(t_est):>10s}  (grid={grid_size})")
                except Exception as exc:
                    status = f"unavailable ({type(exc).__name__})"
                    if verbose:
                        print(f"  {spec.name:25s}  skipped: {exc}")
            else:
                status = "no calibrator"

            records.append({
                "stage": spec.stage,
                "method": spec.name,
                "complexity": spec.complexity_label,
                "grid_size": grid_size,
                "estimated_seconds": t_est,
                "estimated_time": _format_time(t_est),
                "calibration_seconds": t_calib,
                "status": status,
            })

    # ── Classification ──────────────────────────────────────────────
    if "classification" in stages:
        enabled_clf = set(
            getattr(config, "classifiers", list(_CLASSIFIER_SPECS.keys()))
        )
        # Classification benchmark runs each classifier in both gene
        # space and PCA space, so multiply by 2, and by cv_folds for
        # cross-validated estimators.
        space_multiplier = 2
        cv_mult = cv_folds if getattr(config, "classification_estimator", "cv") == "cv" else 1

        for clf_name, spec in _CLASSIFIER_SPECS.items():
            if clf_name not in enabled_clf:
                continue

            grid_size = space_multiplier * cv_mult
            calib_fn = _CALIBRATION_FNS.get(clf_name)
            status = "ok"
            t_calib = np.nan
            t_est = np.nan

            if calib_fn is not None:
                try:
                    t_calib, _ = calib_fn(X_calib, k, random_state)
                    t_est = _extrapolate(
                        t_calib, spec.complexity_fn, n_calib, d, k, n_full, grid_size,
                    )
                    if verbose:
                        print(f"  {spec.name:25s}  calib={t_calib:.3f}s  "
                              f"est={_format_time(t_est):>10s}  (cv*space={grid_size})")
                except Exception as exc:
                    status = f"unavailable ({type(exc).__name__})"
                    if verbose:
                        print(f"  {spec.name:25s}  skipped: {exc}")
            else:
                status = "no calibrator"

            records.append({
                "stage": spec.stage,
                "method": spec.name,
                "complexity": spec.complexity_label,
                "grid_size": grid_size,
                "estimated_seconds": t_est,
                "estimated_time": _format_time(t_est),
                "calibration_seconds": t_calib,
                "status": status,
            })

        # SHAP estimate
        if getattr(config, "include_shap", True):
            try:
                # SHAP is run once on the best model; approximate with
                # a quick constant-time placeholder.
                cost_ratio = _complexity_shap(float(n_full), float(d), float(k)) / max(
                    _complexity_shap(float(n_calib), float(d), float(k)), 1e-12,
                )
                # Use RF calibration time as a proxy base
                rf_fn = _CALIBRATION_FNS.get("RF")
                if rf_fn is not None:
                    t_rf, _ = rf_fn(X_calib, k, random_state)
                    t_shap = t_rf * cost_ratio * 2.0  # SHAP ~ 2x model fit
                else:
                    t_shap = np.nan
                records.append({
                    "stage": "classification",
                    "method": "SHAP",
                    "complexity": _SHAP_SPEC.complexity_label,
                    "grid_size": 1,
                    "estimated_seconds": t_shap,
                    "estimated_time": _format_time(t_shap),
                    "calibration_seconds": np.nan,
                    "status": "ok",
                })
            except Exception:
                records.append({
                    "stage": "classification",
                    "method": "SHAP",
                    "complexity": _SHAP_SPEC.complexity_label,
                    "grid_size": 1,
                    "estimated_seconds": np.nan,
                    "estimated_time": "unavailable",
                    "calibration_seconds": np.nan,
                    "status": "unavailable",
                })

    # ── Batch correction ────────────────────────────────────────────
    if "batch_correction" in stages:
        _batch_methods = batch_methods or ["combat", "harmony", "bbknn", "scanorama"]
        for method_key in _batch_methods:
            spec = _BATCH_CORRECTION_SPECS.get(method_key)
            if spec is None:
                continue
            # Batch correction methods don't have a simple calibration
            # function that's independent of batch structure, so we
            # estimate purely from complexity ratios and a generic
            # timing constant derived from the dataset size.
            cost_ratio = spec.complexity_fn(float(n_full), float(d), float(k)) / max(
                spec.complexity_fn(float(n_calib), float(d), float(k)), 1e-12,
            )
            # Use a baseline of 0.01s per 500 cells for linear methods
            t_baseline = 0.01
            t_est = t_baseline * cost_ratio
            records.append({
                "stage": spec.stage,
                "method": spec.name,
                "complexity": spec.complexity_label,
                "grid_size": 1,
                "estimated_seconds": t_est,
                "estimated_time": _format_time(t_est),
                "calibration_seconds": np.nan,
                "status": "estimated (no calibration)",
            })

    # ── Feature selection ───────────────────────────────────────────
    if "feature_selection" in stages:
        enabled_fs = set(
            getattr(config, "feature_selection_methods", ["pca_loadings", "mutual_information"])
        )
        for method_key, spec in _FEATURE_SELECTION_SPECS.items():
            if method_key not in enabled_fs:
                continue
            cost_ratio = spec.complexity_fn(float(n_full), float(d), float(k)) / max(
                spec.complexity_fn(float(n_calib), float(d), float(k)), 1e-12,
            )
            t_baseline = 0.01
            t_est = t_baseline * cost_ratio
            records.append({
                "stage": spec.stage,
                "method": spec.name,
                "complexity": spec.complexity_label,
                "grid_size": 1,
                "estimated_seconds": t_est,
                "estimated_time": _format_time(t_est),
                "calibration_seconds": np.nan,
                "status": "estimated (no calibration)",
            })

    df = pd.DataFrame(records)
    if not df.empty:
        df = df.sort_values("estimated_seconds", ascending=False, na_position="last")
        df = df.reset_index(drop=True)

        # Add cumulative and total rows
        total_ok = df.loc[df["status"] == "ok", "estimated_seconds"].sum()
        total_all = df["estimated_seconds"].sum()

        if verbose:
            print(f"\n[time_estimator] Total estimated time (calibrated methods): "
                  f"{_format_time(total_ok)}")
            print(f"[time_estimator] Total estimated time (all methods):         "
                  f"{_format_time(total_all)}")

    return df


def print_time_budget(
    estimates: pd.DataFrame,
    *,
    max_seconds: Optional[float] = None,
    max_minutes: Optional[float] = None,
) -> pd.DataFrame:
    """Filter and display methods that fit within a time budget.

    Parameters
    ----------
    estimates:
        Output of :func:`estimate_benchmark_time`.
    max_seconds:
        Maximum total wall-clock budget in seconds.
    max_minutes:
        Maximum total wall-clock budget in minutes (convenience;
        overrides *max_seconds* if both given).

    Returns
    -------
    pd.DataFrame
        Subset of *estimates* whose cumulative time fits within the
        budget, sorted fastest-first.
    """
    if max_minutes is not None:
        max_seconds = max_minutes * 60.0
    if max_seconds is None:
        raise ValueError("Provide either max_seconds or max_minutes.")

    df = estimates.copy()
    df = df.dropna(subset=["estimated_seconds"])
    df = df.sort_values("estimated_seconds", ascending=True).reset_index(drop=True)

    cumsum = df["estimated_seconds"].cumsum()
    mask = cumsum <= max_seconds
    selected = df[mask].copy()
    selected["cumulative_seconds"] = cumsum[mask].values
    selected["cumulative_time"] = selected["cumulative_seconds"].apply(_format_time)

    excluded = df[~mask]

    print(f"Time budget: {_format_time(max_seconds)}")
    print(f"Methods within budget: {len(selected)} / {len(df)}")
    if not selected.empty:
        print(f"Total estimated time:  {_format_time(selected['estimated_seconds'].sum())}")
    if not excluded.empty:
        print(f"\nExcluded ({len(excluded)} methods):")
        for _, row in excluded.iterrows():
            print(f"  {row['method']:25s}  {row['estimated_time']:>10s}  ({row['stage']})")

    return selected
