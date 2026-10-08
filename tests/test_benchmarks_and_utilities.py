"""End-to-end runs of the benchmarks, the benchmarking utilities and the remaining array-level modules."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import scintilla as si


@pytest.fixture
def small(adata_logged):
    """120 cells x 200 genes (30 per type) keeping the PCA."""
    keep = np.concatenate([np.flatnonzero(adata_logged.obs.cell_type == t)[:30] for t in sorted(adata_logged.obs.cell_type.unique())])
    return adata_logged[keep].copy()


# ── benchmarking utilities ──────────────────────────────────────────────


def test_profile_method_reports_time_and_memory():
    result = si.benchmark.profile_method(lambda n: np.zeros(n).sum())(1_000_000)
    assert result.result == 0 and result.elapsed_seconds >= 0 and result.peak_memory_mb > 1


def test_seed_stability_test_is_consistent_for_a_deterministic_method(small):
    def method(adata, random_state):
        return si.tl.kmeans(adata.obsm["X_pca"], n_clusters=4, random_state=random_state)[0]

    def metric(labels):
        return float(si.stats.adjusted_rand_index(small.obs.cell_type.to_numpy(), labels))

    table = si.benchmark.seed_stability_test(
        method, small, metric, n_seeds=6, bootstrap_ci=True, n_bootstrap=200, compute_icc=True
    )
    assert len(table) == 6 and table["metric_value"].min() > 0.9
    assert "icc" in table.attrs and 0.0 <= table.attrs["icc"] <= 1.0
    assert table.attrs["ci_low"] <= table.attrs["ci_high"]


def test_scalability_sweep_profiles_growing_subsets(small):
    table = si.benchmark.scalability_sweep(
        lambda a: a.X.sum(), small, fractions=[0.25, 0.5, 1.0], n_seeds=2, bootstrap_ci=True, n_bootstrap=50
    )
    assert sorted(table["n_cells"].unique()) == [30, 60, 120]
    assert {"elapsed_seconds", "peak_memory_mb"} <= set(table.columns)


def test_pairwise_method_comparison_permutation_and_mcnemar():
    rng = np.random.default_rng(0)
    results = pd.DataFrame(
        {"method": ["a"] * 8 + ["b"] * 8, "score": np.r_[rng.normal(0.9, 0.01, 8), rng.normal(0.7, 0.01, 8)]}
    )
    out = si.benchmark.pairwise_method_comparison(results, "score", n_permutations=200)
    assert out.loc[0, "significant"] and out.loc[0, "p_value"] < 0.05
    y = np.array([0, 1] * 20)
    preds = {"a": y.copy(), "b": np.where(np.arange(40) < 10, 1 - y, y)}
    mc = si.benchmark.pairwise_method_comparison(
        pd.DataFrame({"method": ["a", "b"], "score": [1.0, 0.75]}), "score", y_true=y, predictions=preds, test="mcnemar"
    )
    assert len(mc) == 1 and mc.loc[0, "p_value"] < 0.05


def test_benchmark_report_collects_and_exports(tmp_path):
    report = si.benchmark.BenchmarkReport()
    report.add_result("clustering", "kmeans", {"ari": 0.9})
    report.add_result("clustering", "leiden", {"ari": 0.95})
    report.add_result("classification", "rf", {"accuracy": 0.8})
    assert list(report.get_leaderboard("clustering")["method"]) == ["leiden", "kmeans"]
    report.export(str(tmp_path))
    assert len(list(tmp_path.glob("*.csv"))) >= 1 and len(report.summary_table()) == 3


def test_time_estimator_extrapolates_and_filters_by_budget(small, capsys):
    config = si.AnalysisConfig.fast().copy(clustering_methods=["kmeans"], classifiers=["LogReg"])
    table = si.benchmark.estimate_benchmark_time(
        small, config, stages=["clustering", "classification"], calibration_cells=60, random_state=0
    )
    assert {"stage", "method", "estimated_seconds", "status"} <= set(table.columns) and len(table) > 0
    kept = si.benchmark.print_time_budget(table, max_minutes=10_000)
    assert len(kept) <= len(table) and "Time budget" in capsys.readouterr().out
    with pytest.raises(ValueError, match="max_seconds or max_minutes"):
        si.benchmark.print_time_budget(table)


# ── benchmarks ──────────────────────────────────────────────────────────


@pytest.fixture
def two_models(monkeypatch):
    from scintilla.classification import benchmark

    monkeypatch.setattr(
        benchmark, "_MODEL_FNS",
        {"LogReg": benchmark._MODEL_FNS["LogReg"], "kNN": benchmark._MODEL_FNS["kNN"]},
    )


@pytest.mark.parametrize("estimator", ["cv", "holdout", "bootstrap_632plus"])
def test_classifier_benchmark_all_estimators(small, two_models, estimator):
    table = si.benchmark.benchmark_classifiers(
        small, "cell_type", estimator=estimator, cv_folds=3, n_bootstrap=5, n_pca_comps=5, random_state=0
    )
    assert set(table["model"]) == {"LogReg", "kNN"} and set(table["space"]) == {"Gene", "PCA"}
    assert (table["status"] == "ok").all() and table["accuracy"].min() > 0.8
    assert si.pl.classification_benchmark(table).axes


def test_classifier_benchmark_with_bootstrap_intervals(small, two_models):
    table = si.benchmark.benchmark_classifiers(
        small, "cell_type", estimator="cv", cv_folds=3, bootstrap_ci=True, n_bootstrap=20, random_state=0
    )
    assert len(table) == 4


def test_clustering_benchmark_runs_every_algorithm_family(small):
    pytest.importorskip("leidenalg")
    config = si.AnalysisConfig.default().copy(
        clustering_methods=["kmeans", "hierarchical", "dbscan", "leiden", "louvain", "spectral", "consensus"],
        leiden_resolutions=[0.5], louvain_resolutions=[0.5], spectral_n_clusters_range=[4],
    )
    table, labels = si.benchmark.benchmark_clustering(small, "cell_type", config=config, random_state=0, n_jobs=1)
    assert {"KMeans", "Hierarchical", "DBSCAN", "Leiden", "Louvain", "Spectral"} <= set(table["method"])
    assert table["ari"].max() > 0.95 and len(labels) == (table["status"] == "ok").sum()


def test_clustering_benchmark_parallel_and_adaptive_resolution(small):
    pytest.importorskip("leidenalg")
    config = si.AnalysisConfig.fast().copy(clustering_methods=["kmeans", "leiden"], leiden_resolutions=[0.3, 1.0])
    table, _ = si.benchmark.benchmark_clustering(
        small, "cell_type", config=config, n_jobs=2, adaptive_resolution=True, random_state=0
    )
    assert "Leiden" in set(table["method"])
    table2, _ = si.benchmark.benchmark_clustering(
        small, "cell_type", config=config, resolution_selection="nvi_stability", random_state=0
    )
    assert len(table2) > 0


def test_feature_selection_benchmark_and_hvg_sensitivity(small):
    table = si.benchmark.benchmark_feature_selection(small, "cell_type", n_features=20, random_state=0)
    assert set(table["method"]) == {"pca_loadings", "mutual_information"} and table["accuracy"].min() > 0.8
    pytest.importorskip("leidenalg")
    sens = si.benchmark.hvg_sensitivity_analysis(small, "cell_type", n_top_genes_values=[50, 100], random_state=0)
    assert len(sens) == 2


def test_transformation_benchmark_ranks_methods_on_raw_counts(adata_dense):
    small = adata_dense[::3].copy()
    table, best, best_adata = si.benchmark.benchmark_transformations(
        small, transformations={k: v for k, v in si.pp.TRANSFORM_REGISTRY.items() if k in {"log_shift_size_factor", "arcsinh_transform"}},
        n_pca_components=5, max_k=3, cell_type_col="cell_type", random_state=0,
    )
    assert best in {"log_shift_size_factor", "arcsinh_transform"} and best_adata.shape == small.shape
    assert set(table["transform"]) == {"log_shift_size_factor", "arcsinh_transform"} and set(table["status"]) == {"ok"}


def test_de_benchmark_lists_every_method(adata_dense):
    adata_dense.obs["condition"] = np.where(adata_dense.obs.cell_type.isin(["type0", "type1"]), "A", "B")
    table = si.benchmark.benchmark_de(adata_dense, "condition", "A", "B")
    assert len(table) >= 3


# ── array-level modules not covered elsewhere ───────────────────────────


def test_ora_finds_the_enriched_set():
    background = [f"g{i}" for i in range(100)]
    table = si.tl.ora_test(background[:10], background, {"hit": background[:12], "miss": background[50:70]})
    row = table.set_index("gene_set")
    assert row.loc["hit", "p_value"] < 0.001 and row.loc["miss", "p_value"] > 0.5
    assert bool(row.loc["hit", "significant"]) and row.loc["hit", "n_overlap"] == 10


def test_de_table_helpers_annotate_and_filter(adata_dense):
    table = si.tl.wilcoxon(adata_dense, "cell_type", "type0", "type1")
    annotated = si.tl.volcano_plot_data(table)
    assert set(annotated["regulation"]) <= {"up", "down", "not_significant"}
    assert annotated.query("significant")["p_adjusted"].lt(0.05).all()
    kept = si.tl.filter_de_genes(table, effect_size_col="cliffs_delta")
    assert len(kept) <= len(table)


def test_embedding_and_patient_level_metrics():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(60, 10))
    assert si.stats.trustworthiness(X, X[:, :5]) > 0.6
    assert si.stats.knn_preservation(X, X) == pytest.approx(1.0)
    preds = pd.DataFrame(
        {"patient": ["p1"] * 3 + ["p2"] * 3, "true": ["a"] * 3 + ["b"] * 3, "predicted": ["a", "a", "b", "b", "b", "b"]}
    )
    out = si.stats.aggregate_predictions_to_patient(preds, "patient", "true", "predicted")
    assert out.set_index("patient")["predicted_label"].to_dict() == {"p1": "a", "p2": "b"}
    result = si.stats.patient_confusion_matrix(["a", "b"], ["a", "b"])
    assert result["accuracy"] == 1.0 and result["n_patients"] == 2 and np.trace(result["confusion_matrix"]) == 2


def test_classification_diagnostics(small):
    from scintilla.classification.diagnostics import (
        check_normality_for_classifier,
        covariance_homogeneity_test,
        influential_cells,
    )

    assert isinstance(check_normality_for_classifier(small, "cell_type", random_state=0), (bool, np.bool_))
    result = covariance_homogeneity_test(small[:, :5].copy(), "cell_type")
    assert result is not None
    y = np.array([0, 1] * 30)
    pred = np.where(np.arange(60) % 7 == 0, 1 - y, y)
    out = influential_cells(y, pred, lambda a, b: float((a == b).mean()), B=50, seed=0)
    assert out is not None


def test_per_gene_group_tests(adata_dense):
    genes = [f"gene{i}" for i in range(5)]
    anova = si.stats.anova_per_gene(adata_dense, genes, "cell_type", correction="fdr_bh")
    assert len(anova) == 5 and anova["significant"].all()
    kruskal = si.stats.kruskal_per_gene(adata_dense, genes, "cell_type")
    assert len(kruskal) == 5


def test_similarity_assembly_clusters_patients_per_cell_type(adata_dense):
    from scintilla.clustering.similarity_assembly import build_per_celltype_clustering

    adata_dense.obs["patient"] = [f"p{i % 6}" for i in range(adata_dense.n_obs)]
    result = build_per_celltype_clustering(adata_dense, "patient", "cell_type", n_clusters=2)
    assert set(result) <= {f"type{i}" for i in range(4)} and all(len(v) == 6 for v in result.values())
