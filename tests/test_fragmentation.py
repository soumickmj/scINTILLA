"""Executable contracts for the label fragmentation component."""

import anndata as ad
import numpy as np
import pandas as pd
import pytest

from scintilla.classification.visualise import compute_label_quality_score


def _fake_benchmark(clusters):
    """Stand-in for benchmark_clustering_methods returning one fixed clustering."""
    def fake(*args, **kwargs):
        results = pd.DataFrame({"method": ["fixed"], "params": ["p"], "ari": [1.0]})
        return results, {"fixed_p": np.asarray(clusters)}, None
    return fake


def _cluster(monkeypatch, labels, clusters):
    from scintilla.clustering import run

    monkeypatch.setattr(run, "benchmark_clustering_methods", _fake_benchmark(clusters))
    rng = np.random.default_rng(0)
    adata = ad.AnnData(X=rng.normal(size=(len(labels), 3)), obs=pd.DataFrame({"cell_type": labels}))
    adata.obsm["X_emb"] = np.asarray(adata.X)
    run.unsupervised_analysis(adata, cell_type_col="cell_type", run_pca_first=False,
                              use_rep="X_emb", verbose=False)
    return adata


def test_fragmentation_label_mean_is_gini_simpson_impurity(monkeypatch):
    """A label split 60/40 over two clusters has impurity 1 - 0.6^2 - 0.4^2 = 0.48."""
    labels = ["a"] * 60 + ["b"] * 60
    clusters = ["0"] * 36 + ["1"] * 24 + ["2"] * 60
    adata = _cluster(monkeypatch, labels, clusters)

    per_label = adata.obs.groupby("cell_type")["scintilla_top1_fragmentation"].mean()

    assert np.isclose(per_label["a"], 0.48)
    assert np.isclose(per_label["b"], 0.0)


def test_fragmentation_flags_merged_label_that_confusion_misses(monkeypatch):
    """A label covering two whole clusters is invisible to confusion but fragmented."""
    labels = ["merged"] * 120 + ["c"] * 60 + ["d"] * 60
    clusters = ["0"] * 60 + ["1"] * 60 + ["2"] * 60 + ["3"] * 60
    adata = _cluster(monkeypatch, labels, clusters)

    per_label = adata.obs.groupby("cell_type")[
        ["scintilla_top1_confusion", "scintilla_top1_fragmentation"]].mean()

    assert (per_label["scintilla_top1_confusion"] == 0).all()
    assert per_label.loc["merged", "scintilla_top1_fragmentation"] == 0.5
    quality = compute_label_quality_score(adata, force=True)
    assert quality.idxmin() == "merged"


def _metrics_adata():
    adata = ad.AnnData(X=np.zeros((9, 1)), obs=pd.DataFrame({"cell_type": list("aaabbbccc")}))
    adata.obs["scintilla_top1_confusion"] = [0.1, 0.1, 0.1, 0.5, 0.5, 0.5, 0.2, 0.2, 0.2]
    adata.obs["pred_agreement"] = [1.0, 1.0, 1.0, 0.6, 0.6, 0.6, 0.9, 0.9, 0.9]
    return adata


def test_zero_fragmentation_weight_reproduces_original_score():
    """Weight 0 must equal the score computed before fragmentation existed."""
    before = compute_label_quality_score(_metrics_adata(), force=True)
    adata = _metrics_adata()
    adata.obs["scintilla_top1_fragmentation"] = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.5, 0.5]

    after = compute_label_quality_score(adata, force=True, fragmentation_weight=0.0)
    with_frag = compute_label_quality_score(adata, force=True)

    pd.testing.assert_series_equal(before.sort_index(), after.sort_index())
    assert with_frag["c"] < after["c"]


@pytest.mark.parametrize("n_cells", [2, 20, 50, 51])
def test_confusion_handles_datasets_at_neighbor_count_boundary(monkeypatch, n_cells):
    labels = ["a"] * (n_cells // 2) + ["b"] * (n_cells - n_cells // 2)
    adata = _cluster(monkeypatch, labels, labels)
    assert (adata.obs.scintilla_top1_confusion == 0.0).all()
    assert (adata.obs.scintilla_top1_fragmentation == 0.0).all()


def test_clustering_rerun_replaces_previous_top_rank_metrics(monkeypatch):
    from scintilla.clustering import run

    labels = ["a"] * 60 + ["b"] * 60
    adata = _cluster(monkeypatch, labels, labels)
    adata.obs["scintilla_top2_old|p"] = pd.Categorical(["0"] * 120)
    adata.obs["scintilla_top2_confusion"] = 0.9
    adata.obs["scintilla_top2_fragmentation"] = 0.8
    adata.obs["scintilla_label_quality"] = 0.1
    adata.obs["user_annotation"] = "keep"
    run.unsupervised_analysis(adata, cell_type_col="cell_type", run_pca_first=False,
                              use_rep="X_emb", verbose=False)
    assert not any(c.startswith("scintilla_top2_") for c in adata.obs)
    assert "scintilla_label_quality" not in adata.obs
    assert (adata.obs.user_annotation == "keep").all()


def test_failed_clustering_rerun_does_not_leave_old_label_metrics(monkeypatch):
    from scintilla.clustering import run

    adata = _cluster(monkeypatch, ["a"] * 60 + ["b"] * 60, ["0"] * 60 + ["1"] * 60)
    def failed_benchmark(*args, **kwargs):
        return pd.DataFrame({"method": ["failed"], "params": ["p"], "ari": [np.nan]}), {}, None
    monkeypatch.setattr(run, "benchmark_clustering_methods", failed_benchmark)
    result = run.unsupervised_analysis(adata, cell_type_col="cell_type", run_pca_first=False,
                                       use_rep="X_emb", verbose=False)
    assert result["best_method"] is None
    assert not any(c.startswith("scintilla_top") for c in adata.obs)
    assert "scintilla_cluster" not in adata.obs
