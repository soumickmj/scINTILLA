"""Regression tests for the follow-up correctness and contract fixes.

Each test here pins behaviour that was wrong or undefined after the initial
Phase-0 pass: silent metric failures, a mis-oriented Harmony embedding, an
unvalidated consensus method list, a columnless batch leaderboard, and
optional classifiers that reported "not installed" as a genuine failure.
"""

import sys
import types
import warnings

import anndata as ad
import numpy as np
import pandas as pd
import pytest


# ── Consensus clustering: validated methods and a typed failures channel ──


def test_consensus_rejects_unknown_method_names() -> None:
    """Catch a mistyped method silently contributing nothing to the co-matrix."""
    from scintilla.clustering.consensus import consensus_clustering

    adata = ad.AnnData(X=np.arange(24, dtype=float).reshape(6, 4))
    with pytest.raises(ValueError, match="Unknown consensus method"):
        consensus_clustering(adata, methods=["kmenas"])
    with pytest.raises(ValueError, match="Unknown consensus method"):
        consensus_clustering(adata, methods=["failures"])


def test_consensus_stability_scores_are_float_per_requested_method() -> None:
    """Catch per-method stability values drifting away from plain floats."""
    from scintilla.clustering.consensus import consensus_clustering

    adata = ad.AnnData(X=np.arange(24, dtype=float).reshape(6, 4))
    _, _, stability = consensus_clustering(
        adata, methods=["kmeans"], n_runs_per_method=3
    )

    assert set(stability) == {"kmeans"}
    assert isinstance(stability["kmeans"], float)


def test_consensus_failures_stay_in_the_reserved_typed_channel(monkeypatch) -> None:
    """Catch failure detail leaking into the per-method float mapping."""
    from scintilla.clustering import kmeans
    from scintilla.clustering.consensus import (
        _FAILURES_KEY,
        consensus_clustering,
    )

    def flaky_kmeans(X, n_clusters, **_kwargs):
        if n_clusters == 2:
            raise RuntimeError("k=2 unavailable")
        return np.array([0, 0, 1, 1, 2, 2]), None, {}

    monkeypatch.setattr(kmeans, "kmeans_clustering", flaky_kmeans)
    _, _, stability = consensus_clustering(
        ad.AnnData(X=np.arange(24, dtype=float).reshape(6, 4)),
        methods=["kmeans"],
        n_runs_per_method=2,
    )

    assert _FAILURES_KEY == "failures"
    assert stability[_FAILURES_KEY] == {"kmeans": "k=2 unavailable"}
    method_scores = {k: v for k, v in stability.items() if k != _FAILURES_KEY}
    assert all(isinstance(value, float) for value in method_scores.values())


# ── Classification metrics: expected failures visible, others propagated ──


def test_scalar_metric_failures_warn_by_name_instead_of_silent_nan() -> None:
    """Catch MCC/kappa/balanced-accuracy errors collapsing into unexplained NaN."""
    from scintilla.evaluation import classification_metrics as cm

    def broken(*_args, **_kwargs):
        raise ValueError("metric unavailable")

    original = cm.matthews_corrcoef
    cm.matthews_corrcoef = broken
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            metrics = cm.calculate_metrics(
                np.array([0, 0, 1, 1]), np.array([0, 1, 1, 1])
            )
    finally:
        cm.matthews_corrcoef = original

    assert np.isnan(metrics["mcc"])
    assert [str(item.message) for item in caught] == [
        "mcc could not be computed: metric unavailable"
    ]


def test_unexpected_scalar_metric_failures_propagate() -> None:
    """Catch genuine bugs being laundered into a NaN metric."""
    from scintilla.evaluation import classification_metrics as cm

    def broken(*_args, **_kwargs):
        raise RuntimeError("internal bug")

    original = cm.cohen_kappa_score
    cm.cohen_kappa_score = broken
    try:
        with pytest.raises(RuntimeError, match="internal bug"):
            cm.calculate_metrics(np.array([0, 0, 1, 1]), np.array([0, 1, 1, 1]))
    finally:
        cm.cohen_kappa_score = original


# ── Harmony: embedding orientation must follow the caller's cell count ──


def _harmony_stub(z_corr):
    """Install a fake harmonypy module whose run_harmony returns *z_corr*."""
    module = types.ModuleType("harmonypy")

    def run_harmony(data_mat, meta_data, vars_use, **_kwargs):
        return types.SimpleNamespace(Z_corr=z_corr)

    module.run_harmony = run_harmony
    return module


def _batched_adata(n_obs=6, n_vars=4):
    return ad.AnnData(
        X=np.arange(n_obs * n_vars, dtype=float).reshape(n_obs, n_vars),
        obs=pd.DataFrame(
            {"batch": ["b1", "b2"] * (n_obs // 2)},
            index=[f"c{i}" for i in range(n_obs)],
        ),
    )


@pytest.mark.parametrize("transposed", [False, True])
def test_harmony_accepts_both_upstream_embedding_orientations(
    monkeypatch, transposed
) -> None:
    """Catch `Z_corr.T` producing an obsm value that does not match n_obs."""
    from scintilla.batch_correction import harmony as harmony_mod

    adata = _batched_adata()
    adata.obsm["X_pca"] = np.arange(adata.n_obs * 2, dtype=float).reshape(
        adata.n_obs, 2
    )
    cells_by_pcs = np.arange(adata.n_obs * 2, dtype=float).reshape(adata.n_obs, 2)
    z_corr = cells_by_pcs.T if transposed else cells_by_pcs
    monkeypatch.setitem(sys.modules, "harmonypy", _harmony_stub(z_corr))

    corrected = harmony_mod.harmony_correct(adata, "batch", n_components=2)

    np.testing.assert_array_equal(corrected.obsm["X_pca_harmony"], cells_by_pcs)


def test_harmony_rejects_an_embedding_matching_neither_orientation(
    monkeypatch,
) -> None:
    """Catch a silent shape mismatch being pushed into obsm."""
    from scintilla.batch_correction import harmony as harmony_mod

    adata = _batched_adata()
    adata.obsm["X_pca"] = np.zeros((adata.n_obs, 2))
    monkeypatch.setitem(sys.modules, "harmonypy", _harmony_stub(np.zeros((3, 5))))

    with pytest.raises(ValueError, match="unexpected shape"):
        harmony_mod.harmony_correct(adata, "batch", n_components=2)


# ── Batch benchmark: an empty method list must say so ──


def test_batch_benchmark_rejects_an_empty_method_list() -> None:
    """Catch a columnless leaderboard surfacing as KeyError('batch_asw')."""
    from scintilla.batch_correction.benchmark import benchmark_batch_correction

    with pytest.raises(ValueError, match="at least one batch correction method"):
        benchmark_batch_correction(_batched_adata(), "batch", methods=[])


# ── Optional classifiers: absent backend is a skip, not a failure ──


def test_optional_models_are_absent_when_their_backend_is_not_installed(
    monkeypatch,
) -> None:
    """Catch "xgboost is not installed" being reported as a model failure."""
    import importlib

    from scintilla.classification import benchmark as bench

    monkeypatch.setattr(bench, "_optional_backend_available", lambda name: False)
    assert bench._resolve_optional_models() == {}

    monkeypatch.setattr(bench, "_optional_backend_available", lambda name: True)
    resolved = bench._resolve_optional_models()
    assert set(resolved) == {"XGBoost", "LightGBM"}
    importlib.invalidate_caches()


def test_supervised_analysis_omits_uninstalled_optional_models(monkeypatch) -> None:
    """Catch failed rows for optional models the environment cannot provide."""
    from scintilla.classification import benchmark as bench
    from scintilla.classification.run import supervised_analysis

    monkeypatch.setattr(bench, "_optional_backend_available", lambda name: False)
    rng = np.random.default_rng(0)
    X = rng.normal(size=(40, 6))
    labels = np.array(["a"] * 20 + ["b"] * 20)
    X[labels == "b"] += 4.0
    adata = ad.AnnData(
        X=X,
        obs=pd.DataFrame({"cell_type": labels}, index=[f"c{i}" for i in range(40)]),
    )

    result = supervised_analysis(
        adata, verbose=False, include_shap=False, normality=True
    )

    assert "XGBoost" not in result["all_results"]
    assert "LightGBM" not in result["all_results"]


def test_clustering_benchmark_marks_absent_backend_as_skipped(monkeypatch) -> None:
    """Catch an uninstalled optional backend being reported as a failed method."""
    import builtins

    from matplotlib import pyplot as plt
    from scintilla import AnalysisConfig
    from scintilla.clustering import benchmark

    real_import = builtins.__import__

    def no_louvain(name, *args, **kwargs):
        if name == "scintilla.clustering.louvain":
            raise ImportError("No module named 'louvain'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", no_louvain)
    adata = ad.AnnData(
        X=np.arange(24, dtype=float).reshape(6, 4),
        obs=pd.DataFrame(
            {"cell_type": ["a", "a", "a", "b", "b", "b"]},
            index=[f"c{i}" for i in range(6)],
        ),
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        results, _, figure = benchmark.benchmark_clustering_methods(
            adata,
            "cell_type",
            use_rep="X",
            verbose=False,
            config=AnalysisConfig(clustering_methods=["louvain"]),
        )
    plt.close(figure)

    row = results.iloc[0]
    assert row["method"] == "Louvain"
    assert row["status"] == "skipped"
    assert row["failure_reason"] == "No module named 'louvain'"
    assert not [item for item in caught if "Louvain" in str(item.message)]
