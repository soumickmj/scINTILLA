"""Executable contracts for the per-label scINTILLA variant table."""

import anndata as ad
import numpy as np
import pandas as pd
import pytest

from scintilla.classification.label_quality_variants import (
    FROZEN_FUSION, VARIANTS, compute_label_quality_variants, quality_rank, review_ranks)
from scintilla.classification.visualise import compute_label_quality_score


def _adata():
    """Four labels x 5 cells with every column the pipeline writes."""
    rng = np.random.default_rng(0)
    labels = np.repeat(list("abcd"), 5)
    per_label = {
        "scintilla_top1_confusion": [0.1, 0.4, 0.2, 0.0],
        "scintilla_top2_confusion": [0.2, 0.3, 0.1, 0.0],
        "scintilla_top1_fragmentation": [0.0, 0.1, 0.6, 0.2],
        "scintilla_top2_fragmentation": [0.1, 0.0, 0.5, 0.3],
        "pred_agreement": [0.9, 0.5, 0.8, 1.0],
        "pred_entropy": [0.2, 0.9, 0.3, 0.1],
        "pred_avg_confidence": [0.8, 0.4, 0.7, 0.9],
    }
    obs = pd.DataFrame({"cell_type": labels})
    for col, vals in per_label.items():
        obs[col] = np.repeat(vals, 5)
    x = ad.AnnData(X=np.zeros((20, 1)), obs=obs)
    x.obsm["X_pca"] = rng.normal(size=(20, 3)) + np.repeat(np.arange(4), 5)[:, None] * 3
    return x


def test_columns_follow_variant_order_and_one_row_per_label():
    scores = compute_label_quality_variants(_adata())
    assert list(scores.columns) == list(VARIANTS)
    assert sorted(scores.index) == list("abcd")


def test_composites_match_compute_label_quality_score():
    x = _adata()
    scores = compute_label_quality_variants(x)
    orig = compute_label_quality_score(_adata(), force=True, fragmentation_weight=0.0)
    frag = compute_label_quality_score(_adata(), force=True)
    pd.testing.assert_series_equal(scores.scintilla_composite, orig.reindex(scores.index),
                                   check_names=False, check_index_type=False)
    pd.testing.assert_series_equal(scores.scintilla_composite_frag, frag.reindex(scores.index),
                                   check_names=False, check_index_type=False)
    assert "_scintilla_variant_tmp" not in x.obs


def test_ablations_drop_the_right_arms():
    scores = compute_label_quality_variants(_adata())
    # supervised-only ignores confusion/fragmentation: label d has best supervised columns
    assert scores.scintilla_sup_only.idxmax() == "d"
    # unsupervised-only: confusion only; b has the highest confusion
    assert scores.scintilla_unsup_only.idxmin() == "b"
    # confusion + fragmentation: c has by far the most fragmentation
    assert scores.scintilla_fragmentation_mean.idxmin() == "c"
    assert np.isclose(scores.loc["c", "scintilla_fragmentation_mean"], -0.55)
    assert np.isclose(scores.loc["b", "scintilla_confusion_mean"], -0.35)
    assert np.isclose(scores.loc["b", "scintilla_pred_entropy"], -0.9)


def test_fusions_follow_the_v4_formulas():
    sil = pd.Series({"a": 0.5, "b": 0.1, "c": 0.3, "d": 0.4})
    scores = compute_label_quality_variants(_adata(), silhouette=sil)
    r_s, r_g = quality_rank(scores.scintilla_composite), quality_rank(sil)
    pd.testing.assert_series_equal(scores["scintilla_composite__silhouette"],
                                   ((r_s + r_g) / 2).round(12), check_names=False)
    add = FROZEN_FUSION["fusion_add"]
    expected = -(add["intercept"] + add["coef"]["r_s"] * r_s + add["coef"]["r_g"] * r_g)
    pd.testing.assert_series_equal(scores.fusion_add, expected, check_names=False)
    log = FROZEN_FUSION["fusion_log"]
    expected = -(log["intercept"] + log["coef"]["r_s"] * r_s + log["coef"]["r_g"] * r_g
                 + log["coef"]["r_sg"] * r_s * r_g)
    pd.testing.assert_series_equal(scores.fusion_log, expected, check_names=False)


def test_quality_rank_is_midrank_percentile_with_ties_and_nan():
    r = quality_rank(pd.Series([1.0, 2.0, 2.0, np.nan]))
    assert np.allclose(r[:3], [0.5 / 3, 2.0 / 3, 2.0 / 3])
    assert np.isnan(r.iloc[3])


def test_review_rank_one_is_lowest_score():
    scores = compute_label_quality_variants(_adata())
    ranks = review_ranks(scores)
    assert ranks.loc[scores.scintilla_composite.idxmin(), "scintilla_composite"] == 1


def test_missing_fragmentation_columns_raise():
    x = _adata()
    x.obs = x.obs.drop(columns=["scintilla_top1_fragmentation", "scintilla_top2_fragmentation"])
    with pytest.raises(ValueError, match="fragmentation"):
        compute_label_quality_variants(x)


def test_unused_label_categories_do_not_create_phantom_scores():
    x = _adata()
    x.obs["cell_type"] = pd.Categorical(x.obs["cell_type"], categories=list("abcde"))
    scores = compute_label_quality_variants(x)
    assert set(scores.index) == set("abcd")
    assert scores.notna().all().all()


def test_supervised_ablation_ignores_missing_confusion_values():
    x = _adata()
    x.obs.loc[x.obs.cell_type == "a", "scintilla_top1_confusion"] = np.nan
    scores = compute_label_quality_variants(x)
    assert scores.loc["a", "scintilla_sup_only"] == pytest.approx(0.825)


def test_unsupervised_ablation_ignores_missing_supervised_values():
    x = _adata()
    x.obs.loc[x.obs.cell_type == "a", "pred_agreement"] = np.nan
    scores = compute_label_quality_variants(x)
    assert scores.loc["a", "scintilla_unsup_only"] == pytest.approx(13 / 24)


def test_variant_computation_preserves_existing_temporary_column():
    x = _adata()
    x.obs["_scintilla_variant_tmp"] = pd.Categorical(np.tile(["keep", "me"], 10))
    before = x.obs.copy(deep=True)
    compute_label_quality_variants(x)
    pd.testing.assert_frame_equal(x.obs, before)


def test_missing_top1_confusion_reports_pipeline_prerequisite():
    x = _adata()
    del x.obs["scintilla_top1_confusion"]
    with pytest.raises(ValueError, match="scintilla_top1_confusion"):
        compute_label_quality_variants(x)


def test_missing_active_fragmentation_does_not_improve_quality():
    x = _adata()
    for col in ["scintilla_top1_fragmentation", "scintilla_top2_fragmentation"]:
        x.obs[col] = np.nan
    scores = compute_label_quality_variants(x)
    assert scores.scintilla_composite.notna().all()
    assert scores.scintilla_composite_frag.isna().all()


def test_variant_error_preserves_existing_observation_columns():
    x = _adata()
    x.obs["_scintilla_variant_tmp"] = "keep"
    del x.obsm["X_pca"]
    before = x.obs.copy(deep=True)
    with pytest.raises(KeyError, match="X_pca"):
        compute_label_quality_variants(x)
    pd.testing.assert_frame_equal(x.obs, before)


def _run_label_quality_cli(tmp_path, x, *options):
    from scintilla import AnalysisConfig
    from scintilla.cli.main import main

    source = tmp_path / "input.h5ad"
    config = tmp_path / "config.yaml"
    output = tmp_path / "scores.csv"
    saved = tmp_path / "analysed.h5ad"
    x.write_h5ad(source)
    AnalysisConfig(clustering_methods=["kmeans"], classifiers=["LogReg"],
                   verbose=False).to_yaml(config)
    main(["label-quality", str(source), "--config", str(config),
          "--output", str(output), "--save-h5ad", str(saved), *options])
    scores = pd.read_csv(output, index_col=0)
    assert scores.notna().all().all()
    assert (tmp_path / "scores_ranks.csv").exists()
    return ad.read_h5ad(saved)


def test_cli_reruns_clustering_for_unrelated_fragmentation_column(tmp_path):
    x = _adata()
    x.obs = x.obs.drop(columns=["scintilla_top1_fragmentation", "scintilla_top2_fragmentation"])
    x.obs["user_fragmentation"] = 0.3
    saved = _run_label_quality_cli(tmp_path, x)
    assert "scintilla_top1_fragmentation" in saved.obs
    assert (saved.obs.user_fragmentation == 0.3).all()


def test_cli_reruns_clustering_for_unmatched_fragmentation_ranks(tmp_path):
    x = _adata()
    del x.obs["scintilla_top2_fragmentation"]
    saved = _run_label_quality_cli(tmp_path, x)
    conf = {c.replace("_confusion", "") for c in saved.obs if c.startswith("scintilla_top") and c.endswith("_confusion")}
    frag = {c.replace("_fragmentation", "") for c in saved.obs if c.startswith("scintilla_top") and c.endswith("_fragmentation")}
    assert conf == frag


def test_cli_creates_missing_pca_even_with_cached_metrics(tmp_path):
    pytest.importorskip("scanpy")
    x = _adata()
    x = ad.AnnData(X=np.random.default_rng(0).normal(size=(20, 6)), obs=x.obs.copy())
    saved = _run_label_quality_cli(tmp_path, x)
    assert "X_pca" in saved.obsm


def test_cli_creates_output_directories_for_cached_scores(tmp_path):
    from scintilla.cli.main import main

    source = tmp_path / "input.h5ad"
    _adata().write_h5ad(source)
    output = tmp_path / "new" / "scores.csv"
    saved = tmp_path / "other" / "analysed.h5ad"
    main(["label-quality", str(source), "--output", str(output), "--save-h5ad", str(saved)])
    assert output.exists()
    assert output.with_name("scores_ranks.csv").exists()
    assert saved.exists()


def test_cli_analyses_raw_input_without_cached_metrics(tmp_path):
    pytest.importorskip("scanpy")
    labels = np.repeat(list("abcd"), 10)
    X = np.random.default_rng(0).normal(size=(40, 6))
    X += np.repeat(np.arange(4), 10)[:, None] * 3
    x = ad.AnnData(X=X, obs=pd.DataFrame({"cell_type": labels},
                                        index=[f"c{i}" for i in range(40)]))
    saved = _run_label_quality_cli(tmp_path, x)
    assert "X_pca" in saved.obsm
    assert "scintilla_top1_fragmentation" in saved.obs
    assert "pred_agreement" in saved.obs
    assert "pred_entropy" in saved.obs
