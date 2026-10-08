"""The flagship pipelines: copy, key_added, provenance, h5ad round trip, sparse parity."""

from __future__ import annotations

import warnings

import anndata as ad
import numpy as np
import pandas as pd
import pytest

import scintilla as si

pytest.importorskip("leidenalg")


@pytest.fixture
def fast_config():
    return si.AnalysisConfig.fast().copy(
        include_shap=False,
        clustering_methods=["kmeans", "leiden"],
        leiden_resolutions=[0.5],
        classifiers=["LogReg", "kNN"],
        random_seed=0,
    )


@pytest.fixture
def logged(adata_logged):
    return adata_logged


def test_unsupervised_analysis_writes_the_documented_obs_columns(logged, fast_config):
    result = si.tl.unsupervised_analysis(logged, "cell_type", config=fast_config)
    assert set(result) == {"adata", "results_df", "labels_dict", "best_method"}  # no figure any more
    assert result["adata"] is logged
    assert {"scintilla_top1_confusion", "scintilla_top1_fragmentation", "scintilla_cluster"} <= set(logged.obs)
    assert result["results_df"]["status"].isin(["ok", "failed", "skipped"]).all()
    assert logged.uns["scintilla"]["unsupervised"]["params"]["best_method"] == result["best_method"]


def test_unsupervised_analysis_copy_and_key_added(logged, fast_config):
    columns = set(logged.obs)
    result = si.tl.unsupervised_analysis(logged, "cell_type", config=fast_config, copy=True, key_added="my")
    assert result["adata"] is not logged and set(logged.obs) == columns
    assert {"my_top1_confusion", "my_top1_fragmentation", "my_cluster"} <= set(result["adata"].obs)
    assert not [c for c in result["adata"].obs if c.startswith("scintilla_top")]


def test_unsupervised_analysis_rerun_replaces_instead_of_accumulating(logged, fast_config):
    si.tl.unsupervised_analysis(logged, "cell_type", config=fast_config)
    first = sorted(c for c in logged.obs if c.startswith("scintilla_"))
    si.tl.unsupervised_analysis(logged, "cell_type", config=fast_config)
    assert sorted(c for c in logged.obs if c.startswith("scintilla_")) == first


def test_supervised_analysis_key_added_and_copy(logged, fast_config):
    result = si.tl.supervised_analysis(
        logged, "cell_type", config=fast_config, check_consistency=True, copy=True, key_added="cls"
    )
    out = result["adata"]
    assert {"cls_agreement", "cls_entropy", "cls_consensus", "cls_LogReg", "cls_LogReg_confidence"} <= set(out.obs)
    assert not [c for c in logged.obs if c.startswith(("cls_", "pred_"))]
    assert result["best_model_name"] in {"LogReg", "kNN"} and result["all_results"]["LogReg"]["metrics"]["accuracy"] > 0.9


def test_supervised_analysis_config_include_shap_is_honoured(logged, fast_config, monkeypatch):
    from scintilla.classification import run

    called = []
    monkeypatch.setattr(run, "shap_analysis", lambda *a, **k: called.append(1) or (None, None))
    si.tl.supervised_analysis(logged, "cell_type", config=fast_config)  # config says include_shap=False
    assert called == []


def test_a_failing_model_is_reported_not_dropped(logged, fast_config, monkeypatch):
    from scintilla.classification import run

    def boom(*a, **k):
        raise RuntimeError("this model is broken")

    monkeypatch.setattr(run, "knn_classification", boom)
    with pytest.warns(UserWarning, match="kNN failed: this model is broken"):
        result = si.tl.supervised_analysis(logged, "cell_type", config=fast_config)
    assert result["all_results"]["kNN"]["status"] == "failed"
    assert "this model is broken" in result["all_results"]["kNN"]["failure_reason"]


def test_provenance_and_results_survive_an_h5ad_round_trip(logged, fast_config, tmp_path):
    si.pp.pca(logged, n_comps=5, key_added="X_small")
    si.pp.log_cpm_transform(logged)
    si.tl.unsupervised_analysis(logged, "cell_type", config=fast_config)
    path = tmp_path / "out.h5ad"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        si.io.save_anndata(logged, path)
        back = si.io.load_h5ad(path)
    assert back.uns["scintilla"]["X_small"]["params"]["n_comps"] == 5
    assert "log_cpm_transform" in back.layers
    pd.testing.assert_series_equal(back.obs["scintilla_top1_confusion"], logged.obs["scintilla_top1_confusion"])


def test_label_quality_ranks_the_planted_bad_label_last():
    from _synthetic import make_labelled_blobs

    adata = make_labelled_blobs()
    config = si.AnalysisConfig.fast().copy(clustering_methods=["kmeans", "leiden"], leiden_resolutions=[0.5, 1.0])
    scores = si.tl.label_quality(adata, "cell_type", config=config, random_state=0)
    assert list(scores.index) == ["A", "B", "C", "D"]
    assert scores["scintilla_composite"].idxmin() == "D"  # half of D was relabelled A
    assert set(si.tl.review_ranks(scores)["scintilla_composite"].astype(int)) == {1, 2, 3, 4}


def test_label_quality_reuses_existing_arms_unless_asked_to_rerun(adata_logged, fast_config, monkeypatch):
    from scintilla.clustering import run

    first = si.tl.label_quality(adata_logged, "cell_type", config=fast_config)
    calls = []
    monkeypatch.setattr(run, "benchmark_clustering_methods", lambda *a, **k: calls.append(1))
    again = si.tl.label_quality(adata_logged, "cell_type", config=fast_config)
    pd.testing.assert_frame_equal(first, again)
    assert calls == []


def test_label_quality_agrees_between_dense_and_sparse_input(adata_dense, adata_csr, fast_config):
    for a in (adata_dense, adata_csr):
        a.X = a.X.astype(np.float32)
    dense = si.tl.label_quality(adata_dense, "cell_type", config=fast_config)
    sp = si.tl.label_quality(adata_csr, "cell_type", config=fast_config)
    np.testing.assert_allclose(dense.to_numpy(float), sp.to_numpy(float), atol=1e-6)


def test_label_quality_rejects_a_missing_embedding(adata_dense):
    with pytest.raises(KeyError, match="has no 'X_nope'"):
        si.tl.label_quality(adata_dense, "cell_type", use_rep="X_nope")


def test_pipelines_accept_a_dataframe():
    rng = np.random.default_rng(0)
    df = pd.DataFrame(rng.normal(size=(60, 5)), columns=list("abcde"))
    df["target"] = np.repeat(["x", "y", "z"], 20)
    df.iloc[:20, :3] += 4
    result = si.tl.supervised_analysis(df, "target", models=["LogReg"], include_shap=False)
    assert isinstance(result["adata"], ad.AnnData) and result["best_model_name"] == "LogReg"
