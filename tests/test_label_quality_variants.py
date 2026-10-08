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
