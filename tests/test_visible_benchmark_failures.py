"""Regression coverage for visible method failures in benchmarks."""

import warnings
import sys
from types import SimpleNamespace

import anndata as ad
import numpy as np
import pandas as pd
import pytest

from scintilla.preprocessing.benchmark import benchmark_transformations


def _benchmark_data() -> ad.AnnData:
    return ad.AnnData(X=np.arange(48, dtype=float).reshape(12, 4) + 1.0)


def test_transformation_benchmark_keeps_failed_method_row_and_reason() -> None:
    """Catch a benchmark that only prints a failed method and omits its status."""
    def failing_transform(_adata):
        raise RuntimeError("deliberate transform failure")

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        results, _, _ = benchmark_transformations(
            _benchmark_data(),
            transformations={"healthy": lambda value: value.copy(), "broken": failing_transform},
            verbose=False,
        )

    rows = results.set_index("transform")
    assert set(rows.index) == {"healthy", "broken"}
    assert rows.loc["broken", "status"] == "failed"
    assert rows.loc["broken", "failure_reason"] == "deliberate transform failure"
    assert [str(item.message) for item in caught] == [
        "broken failed: deliberate transform failure"
    ]


def test_transformation_benchmark_returns_rows_when_every_method_fails() -> None:
    """Catch all-failed benchmarks that discard their accumulated failure rows."""
    def failing_transform(_adata):
        raise RuntimeError("all transforms fail")

    results, best_name, best_adata = benchmark_transformations(
        _benchmark_data(),
        transformations={"first": failing_transform, "second": failing_transform},
        verbose=False,
    )

    rows = results.set_index("transform")
    assert set(rows.index) == {"first", "second"}
    assert set(rows["status"]) == {"failed"}
    assert set(rows["failure_reason"]) == {"all transforms fail"}
    assert best_name is None
    assert best_adata is None


def test_consensus_records_partial_method_failure_once(monkeypatch) -> None:
    """Catch a successful consensus arm that hides a failed candidate run."""
    from scintilla.clustering import kmeans
    from scintilla.clustering.consensus import consensus_clustering

    def flaky_kmeans(X, n_clusters, **_kwargs):
        if n_clusters == 2:
            raise RuntimeError("k=2 unavailable")
        return np.array([0, 0, 1, 1, 2, 2]), None, {}

    monkeypatch.setattr(kmeans, "kmeans_clustering", flaky_kmeans)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        _, _, stability = consensus_clustering(
            ad.AnnData(X=np.arange(24, dtype=float).reshape(6, 4)),
            methods=["kmeans"],
            n_runs_per_method=2,
        )

    assert stability["failures"] == {"kmeans": "k=2 unavailable"}
    assert [str(item.message) for item in caught] == [
        "kmeans failed: k=2 unavailable"
    ]


def test_consensus_stability_probe_warns_once_per_method(monkeypatch) -> None:
    """Catch duplicate warnings from each failed pairwise stability calculation."""
    from scintilla.clustering.consensus import consensus_clustering
    from sklearn import metrics

    def broken_ari(*_args, **_kwargs):
        raise ValueError("ARI unavailable")

    monkeypatch.setattr(metrics, "adjusted_rand_score", broken_ari)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        consensus_clustering(
            ad.AnnData(X=np.arange(24, dtype=float).reshape(6, 4)),
            methods=["kmeans"],
            n_runs_per_method=3,
        )

    assert [str(item.message) for item in caught] == [
        "kmeans stability score failed: ARI unavailable"
    ]


def test_dot632plus_keeps_optional_estimator_construction_failure(monkeypatch) -> None:
    """Catch optional-estimator imports that abort .632+ before a result row exists."""
    from matplotlib import pyplot as plt
    from scintilla.classification import benchmark

    def missing_optional_estimator(*_args, **_kwargs):
        raise ModuleNotFoundError("No module named 'xgboost'")

    monkeypatch.setattr(benchmark, "_MODEL_FNS", {"XGBoost": object()})
    monkeypatch.setattr(benchmark, "_get_model_instance", missing_optional_estimator)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        results, figure = benchmark._benchmark_632plus(
            np.arange(24, dtype=float).reshape(6, 4),
            np.array([0, 0, 0, 1, 1, 1]),
            use_pca=False,
            n_pca_comps=2,
            verbose=False,
            B=1,
        )
    plt.close(figure)

    row = results.iloc[0]
    assert row["model"] == "XGBoost"
    assert row["status"] == "failed"
    assert row["failure_reason"] == "No module named 'xgboost'"
    assert [str(item.message) for item in caught] == [
        "XGBoost failed: No module named 'xgboost'"
    ]


def test_clustering_benchmark_keeps_failed_grid_point(monkeypatch) -> None:
    """Catch failed clustering parameters disappearing from leaderboard."""
    from matplotlib import pyplot as plt
    from scintilla import AnalysisConfig
    from scintilla.clustering import benchmark

    def partly_failing_kmeans(X, _n_clusters, *, init="k-means++", **_kwargs):
        if init == "random":
            raise ValueError("random initialisation failed")
        return np.array([0, 0, 0, 1, 1, 1]), None, {}

    monkeypatch.setattr(benchmark, "kmeans_clustering", partly_failing_kmeans)
    adata = ad.AnnData(
        X=np.arange(24, dtype=float).reshape(6, 4),
        obs=pd.DataFrame({"cell_type": ["A", "A", "A", "B", "B", "B"]}),
    )

    with pytest.warns(UserWarning, match="KMeans init=random failed"):
        results, _, figure = benchmark.benchmark_clustering_methods(
            adata,
            cell_type_col="cell_type",
            config=AnalysisConfig(clustering_methods=["kmeans"], verbose=False),
        )
    plt.close(figure)

    failed = results.loc[results["params"] == "init=random"].iloc[0]
    assert failed["status"] == "failed"
    assert failed["failure_reason"] == "random initialisation failed"


def test_hdbscan_grid_failure_is_not_reported_as_all_noise(monkeypatch) -> None:
    """Catch failed HDBSCAN fits masquerading as valid all-noise solutions."""
    from scintilla.clustering.hdbscan import hdbscan_clustering

    class BrokenHDBSCAN:
        def __init__(self, **_kwargs):
            pass

        def fit_predict(self, _X):
            raise ValueError("invalid metric")

    monkeypatch.setitem(
        sys.modules,
        "hdbscan",
        SimpleNamespace(HDBSCAN=BrokenHDBSCAN),
    )

    with pytest.warns(UserWarning, match="HDBSCAN grid point failed"):
        results, _ = hdbscan_clustering(
            np.arange(24, dtype=float).reshape(6, 4),
            min_cluster_size_range=[2],
            min_samples_range=[1],
        )

    row = results.iloc[0]
    assert row["status"] == "failed"
    assert row["failure_reason"] == "invalid metric"


def test_hierarchical_similarity_assembly_never_substitutes_kmeans(monkeypatch) -> None:
    """Catch requested hierarchical clustering silently becoming KMeans."""
    from scintilla.clustering import hierarchical
    from scintilla.clustering.similarity_assembly import build_per_celltype_clustering
    from scintilla.preprocessing import aggregation

    pseudo = pd.DataFrame(
        [[1.0, 0.0], [2.0, 0.0], [3.0, 0.0]],
        index=pd.MultiIndex.from_tuples(
            [("p1", "T"), ("p2", "T"), ("p3", "T")],
            names=["patient", "cell_type"],
        ),
    )
    monkeypatch.setattr(
        aggregation,
        "aggregate_by_patient_celltype",
        lambda *_args, **_kwargs: pseudo,
    )
    monkeypatch.setattr(
        hierarchical,
        "hierarchical_sklearn",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            RuntimeError("hierarchical failed")
        ),
    )

    with pytest.raises(RuntimeError, match="hierarchical failed"):
        build_per_celltype_clustering(
            ad.AnnData(X=np.ones((1, 1))),
            patient_col="patient",
            celltype_col="cell_type",
            method="hierarchical",
            n_clusters=2,
        )
