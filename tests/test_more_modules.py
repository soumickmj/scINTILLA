"""Adaptive PCA, post-hoc tests, batch metrics, selection back-ends, remaining plots, integration."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import scintilla as si

# ── adaptive PCA dimensionality ─────────────────────────────────────────


@pytest.mark.parametrize("method", ["gavish_donoho", "marchenko_pastur"])
def test_pca_auto_components_recovers_the_planted_structure(adata_logged, method):
    si.pp.pca(adata_logged, n_comps=30, auto_components=method, key_added="X_auto", random_state=0)
    kept = adata_logged.obsm["X_auto"].shape[1]
    assert 2 <= kept <= 12  # four planted cell types: a handful of informative components
    assert adata_logged.uns["X_auto"]["auto_method"] == method


def test_pca_variance_threshold_and_unknown_auto_method(adata_logged):
    si.pp.pca(adata_logged, n_comps=30, variance_threshold=0.5, random_state=0)
    n = adata_logged.obsm["X_pca"].shape[1]
    assert 2 <= n < 30 and si.pp.cumulative_variance_explained(adata_logged)[n - 1] >= 0.5 - 1e-6
    with pytest.raises(ValueError, match="Unknown auto_components"):
        si.pp.pca(adata_logged, auto_components="bogus")
    fresh = adata_logged.copy()
    del fresh.uns["pca"]
    with pytest.raises(ValueError, match="PCA has not been run"):
        si.pp.cumulative_variance_explained(fresh)


# ── post-hoc and permutation tests ──────────────────────────────────────


def test_dunn_posthoc_returns_a_symmetric_matrix_of_pairwise_p_values(adata_dense):
    pytest.importorskip("scikit_posthocs")
    table = si.stats.dunn_posthoc(adata_dense, "gene0", "cell_type")
    assert table.shape == (4, 4) and np.allclose(table.to_numpy(), table.to_numpy().T)
    assert table.loc["type0", "type1"] < 0.001  # gene0 sits in type0's up-regulated block


def test_box_m_detects_unequal_covariances():
    rng = np.random.default_rng(0)
    import anndata as ad

    X = np.vstack([rng.normal(size=(80, 3)), rng.normal(size=(80, 3)) * np.array([1, 3, 0.3])]).astype(np.float32)
    adata = ad.AnnData(X=X, obs=pd.DataFrame({"g": ["a"] * 80 + ["b"] * 80}, index=[f"c{i}" for i in range(160)]))
    result = si.stats.box_m_test(adata, "g")
    assert result["p_value"] < 0.001


def test_permutation_test_between_two_methods_and_paired_bootstrap():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 200)
    good, bad = np.where(rng.random(200) < 0.95, y, 1 - y), np.where(rng.random(200) < 0.6, y, 1 - y)
    acc = lambda a, b: float((a == b).mean())  # noqa: E731
    perm = si.stats.permutation_test_methods(y, good, bad, acc, n_permutations=300, seed=0)
    boot = si.stats.paired_bootstrap_test(y, good, bad, acc, B=300, seed=0)
    assert perm["p_value"] < 0.02 and boot["p_value"] < 0.02


# ── batch metrics and embeddings ────────────────────────────────────────


def test_batch_mixing_metrics_separate_mixed_from_segregated_data():
    rng = np.random.default_rng(0)
    import anndata as ad

    n = 200
    batch = np.tile(["b1", "b2"], n // 2)
    mixed = rng.normal(size=(n, 5))
    segregated = mixed + np.where(batch == "b2", 8.0, 0.0)[:, None]
    labels = np.repeat(["x", "y"], n // 2)
    obs = pd.DataFrame({"batch": batch, "label": labels}, index=[f"c{i}" for i in range(n)])
    good = ad.AnnData(X=mixed.astype(np.float32), obs=obs.copy())
    bad = ad.AnnData(X=segregated.astype(np.float32), obs=obs.copy())
    for adata in (good, bad):
        adata.obsm["X_pca"] = adata.X.copy()
    assert si.stats.graph_connectivity is not None
    from scintilla.batch_correction.metrics import batch_asw, kbet_score, lisi_score

    assert batch_asw(good, "batch") > batch_asw(bad, "batch")
    assert lisi_score(good, "batch") > lisi_score(bad, "batch") and lisi_score(good, "batch") > 1.7
    assert kbet_score(mixed, batch) > kbet_score(segregated, batch)
    assert si.stats.principal_component_regression(segregated, mixed, batch, random_state=0) is not None


def test_comprehensive_and_bootstrap_clustering_metrics():
    rng = np.random.default_rng(0)
    X = np.vstack([rng.normal(c, 0.3, size=(40, 3)) for c in (0, 3, 6)])
    true = np.repeat([0, 1, 2], 40)
    full = si.stats.comprehensive_clustering_metrics(X, true, true, bootstrap_ci=True, n_bootstrap=50, seed=0)
    assert full["ari"] == pytest.approx(1.0)
    boot = si.stats.bootstrap_clustering_metrics(X, true, true, B=30, seed=0)
    assert set(boot) == {"ari", "nmi", "silhouette"} and boot["ari"] == {"ci_low": 1.0, "ci_high": 1.0}


# ── selection back-ends ─────────────────────────────────────────────────


def test_mrmr_and_boruta_select_informative_genes(adata_logged):
    from scipy import sparse  # noqa: F401

    pytest.importorskip("mrmr")
    X, y = adata_logged.X[:, :80], adata_logged.obs.cell_type.to_numpy()
    idx, _ = si.mrmr_selection(X, y, n_features=10, random_state=0)
    assert len(idx) == 10 and (np.asarray(idx) < 80).all()


def test_boruta_marks_planted_genes(adata_logged):
    pytest.importorskip("boruta")
    mask, importances = si.boruta_selection(adata_logged.X[:, :60], adata_logged.obs.cell_type.to_numpy(), max_iter=15, random_state=0)
    assert mask.sum() >= 40 and len(importances) == 60


# ── remaining plots ─────────────────────────────────────────────────────


def test_gene_and_transformation_plots(adata_logged):
    genes = [f"gene{i}" for i in range(4)]
    assert si.pl.boxplot_genes_by_group(adata_logged, genes, "cell_type").axes
    assert si.pl.expression_heatmap(adata_logged, genes, "cell_type").axes
    before = adata_logged.copy()
    assert si.pl.before_after_distribution(before, adata_logged, "identity", n_genes=3).axes


def test_benchmark_summary_plots():
    df = pd.DataFrame({"method": ["a", "b", "c"], "x": [0.9, 0.5, 0.7], "y": [0.2, 0.8, 0.6]})
    assert si.pl.benchmark_heatmap(df, "method", ["x", "y"]).axes
    assert si.pl.radar_chart({"ari": 0.9, "ami": 0.7, "silhouette": 0.4}, "radar").axes


def test_sankey_and_batch_comparison_plots(adata_logged):
    pytest.importorskip("plotly")
    assert si.pl.sankey(adata_logged.obs.cell_type.to_numpy(), adata_logged.obs.batch.to_numpy()) is not None
    corrected = adata_logged.copy()
    assert si.pl.batch_correction_comparison(adata_logged, corrected, "batch", "cell_type", random_state=0).axes


# ── integration on a bundled scanpy dataset ─────────────────────────────


@pytest.mark.slow
def test_label_quality_on_pbmc68k_reduced():
    sc = pytest.importorskip("scanpy")
    pytest.importorskip("leidenalg")
    adata = sc.datasets.pbmc68k_reduced()
    scores = si.tl.label_quality(adata, "bulk_labels", fast=True, random_state=0)
    assert set(scores.index) == set(adata.obs["bulk_labels"].astype(str))
    assert scores["scintilla_composite"].between(0, 1).all()
    assert {"fusion_add", "fusion_log", "silhouette"} <= set(scores.columns)
