"""Correctness anchors: the statistics layer against hand-computed values and published formulae."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy import stats as sps
from sklearn import metrics as skm

import scintilla as si

# ── effect sizes ────────────────────────────────────────────────────────


def test_cohens_d_matches_the_textbook_value():
    x1, x2 = np.array([[1.0], [2.0], [3.0]]), np.array([[2.0], [3.0], [4.0], [5.0]])
    # means 2 and 3.5, variances 1 and 5/3, pooled variance (2*1 + 3*5/3)/5 = 1.4
    expected = (2.0 - 3.5) / np.sqrt(1.4)
    np.testing.assert_allclose(si.stats.cohens_d(x1, x2), [expected], rtol=1e-6)


def test_hedges_g_applies_the_small_sample_correction():
    x1, x2 = np.array([[1.0], [2.0], [3.0]]), np.array([[2.0], [3.0], [4.0], [5.0]])
    correction = 1 - 3 / (4 * (3 + 4 - 2) - 1)
    np.testing.assert_allclose(si.stats.hedges_g(x1, x2), si.stats.cohens_d(x1, x2) * correction, rtol=1e-9)


def test_cliffs_delta_counts_dominance_pairs():
    # P(x1 > x2) = 1/9, P(x1 < x2) = 6/9  ->  delta = -5/9
    x1, x2 = np.array([[1.0], [2.0], [3.0]]), np.array([[2.0], [3.0], [4.0]])
    np.testing.assert_allclose(si.stats.cliffs_delta(x1, x2), [-5 / 9])
    assert si.stats.cliffs_delta(x2 + 10, x1)[0] == 1.0


def test_rank_biserial_from_u():
    assert si.stats.rank_biserial(5, 4, 5) == pytest.approx(1 - 2 * 5 / 20)


def test_wilcoxon_effect_sizes_have_a_common_direction(adata_dense):
    table = si.tl.wilcoxon(adata_dense, "cell_type", "type0", "type1")
    up = table[table["log2fc"] > 1]
    assert len(up) > 0 and (up["rank_biserial"] > 0).all() and (up["cliffs_delta"] > 0).all()


# ── multiple testing and aggregation ────────────────────────────────────


def test_correct_pvalues_matches_hand_calculation():
    p = np.array([0.01, 0.02, 0.03, 0.04, 0.05])
    _, bh = si.stats.correct_pvalues(p, "fdr_bh")
    np.testing.assert_allclose(bh, [0.05] * 5)  # p_i * n / rank_i is 0.05 throughout
    _, bonf = si.stats.correct_pvalues(p, "bonferroni")
    np.testing.assert_allclose(bonf, p * 5)


def test_borda_ranks_are_symmetric_for_a_perfect_trade_off():
    scores = pd.DataFrame({"a": [0.9, 0.8, 0.7], "b": [0.5, 0.6, 0.9]}, index=["m1", "m2", "m3"])
    out = si.stats.borda_count(scores)
    assert set(out["borda_score"]) == {4.0}


# ── bootstrap ───────────────────────────────────────────────────────────


def test_bca_interval_agrees_with_the_analytic_t_interval_for_a_mean():
    x = np.random.default_rng(0).normal(5, 2, 200)
    ci = si.stats.bca_bootstrap_ci(x, np.mean, B=2000, seed=0)
    half = sps.t.ppf(0.975, 199) * x.std(ddof=1) / np.sqrt(200)
    assert ci["ci_low"] == pytest.approx(x.mean() - half, abs=0.06)
    assert ci["ci_high"] == pytest.approx(x.mean() + half, abs=0.06)
    assert ci["ci_low"] < ci["point"] < ci["ci_high"]


def test_bootstrap_is_reproducible_for_a_fixed_seed():
    x = np.random.default_rng(1).normal(size=60)
    assert si.stats.bca_bootstrap_ci(x, np.mean, B=300, seed=5) == si.stats.bca_bootstrap_ci(x, np.mean, B=300, seed=5)


def test_bootstrap_metric_ci_brackets_the_point_estimate():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 300)
    pred = np.where(rng.random(300) < 0.8, y, 1 - y)
    ci = si.stats.bootstrap_metric_ci(y, pred, skm.accuracy_score, B=500, seed=0)
    assert ci["point"] == pytest.approx(skm.accuracy_score(y, pred))
    assert ci["ci_low"] < ci["point"] < ci["ci_high"]
    assert 0.74 < ci["point"] < 0.86


# ── data-adaptive dimensionality ────────────────────────────────────────


def _low_rank(rank: int, n: int = 300, p: int = 100, signal: float = 3.0, seed: int = 0):
    rng = np.random.default_rng(seed)
    M = signal * rng.normal(size=(n, rank)) @ rng.normal(size=(rank, p)) + rng.normal(size=(n, p))
    return np.linalg.svd(M - M.mean(axis=0), compute_uv=False), n, p


@pytest.mark.parametrize("rank", [2, 3, 5])
def test_gavish_donoho_recovers_the_planted_rank(rank):
    sv, n, p = _low_rank(rank)
    assert si.stats.gavish_donoho_threshold(sv, n, p) == rank


@pytest.mark.parametrize("rank", [2, 3, 5])
@pytest.mark.parametrize("sigma_method", ["median", "trimmed_mean"])
def test_marchenko_pastur_recovers_the_planted_rank(rank, sigma_method):
    sv, n, p = _low_rank(rank)
    assert si.stats.marchenko_pastur_cutoff(sv, n, p, sigma_method=sigma_method) == rank


def test_gavish_donoho_coefficient_matches_the_published_limit():
    # omega(beta) approximates 2.858 as beta -> 1 (Gavish & Donoho 2014, Fig. 1)
    beta = 1.0
    omega = 0.56 * beta**3 - 0.95 * beta**2 + 1.82 * beta + 1.43
    assert omega == pytest.approx(2.858, abs=0.01)


def test_estimated_dbscan_eps_separates_cluster_density_from_noise():
    rng = np.random.default_rng(0)
    clusters = np.vstack([rng.normal(c, 0.2, size=(100, 2)) for c in ((0, 0), (5, 5), (0, 5))])
    noise = rng.uniform(-3, 8, size=(30, 2))
    X = np.vstack([clusters, noise])
    eps = si.tl.estimate_eps(X, min_samples=5)
    labels, n_clusters, n_noise, _ = si.tl.dbscan(X, eps=eps, min_samples=5)
    assert 0.05 < eps < 2.0
    assert n_clusters == 3 and n_noise >= 15


# ── tests between classifiers ───────────────────────────────────────────


def test_mcnemar_is_uninformative_for_symmetric_disagreement():
    y = np.array([1, 1, 0, 0, 1, 0, 1, 0])
    out = si.stats.mcnemar_test(y, np.array([1, 0, 0, 0, 1, 1, 1, 0]), np.array([1, 1, 0, 1, 1, 0, 0, 0]))
    assert out["p_value"] == pytest.approx(1.0) and out["n_discordant"] == 4


# ── clustering metrics ──────────────────────────────────────────────────


def test_clustering_metrics_on_known_labellings():
    truth = np.repeat([0, 1, 2], 20)
    assert si.stats.adjusted_rand_index(truth, truth) == pytest.approx(1.0)
    assert si.stats.normalised_mutual_info(truth, truth) == pytest.approx(1.0)
    relabelled = (truth + 1) % 3  # a permutation of the names is still a perfect clustering
    assert si.stats.adjusted_rand_index(truth, relabelled) == pytest.approx(1.0)
    chance = np.random.default_rng(0).integers(0, 3, 60)
    assert abs(si.stats.adjusted_rand_index(truth, chance)) < 0.1


def test_clustering_metrics_agree_with_scikit_learn():
    rng = np.random.default_rng(2)
    X = rng.normal(size=(90, 5))
    true, pred = np.repeat([0, 1, 2], 30), rng.integers(0, 3, 90)
    assert si.stats.adjusted_rand_index(true, pred) == pytest.approx(skm.adjusted_rand_score(true, pred))
    assert si.stats.v_measure(true, pred) == pytest.approx(skm.v_measure_score(true, pred))
    assert si.stats.silhouette(X, pred) == pytest.approx(skm.silhouette_score(X, pred))
    assert si.stats.davies_bouldin(X, pred) == pytest.approx(skm.davies_bouldin_score(X, pred))


def test_accuracy_from_mapped_labels_counts_noise_as_wrong():
    true = np.array([0, 0, 0, 1, 1, 1])
    assert si.stats.accuracy_from_mapped_labels(true, np.array([5, 5, 5, 7, 7, 7])) == 1.0
    assert si.stats.accuracy_from_mapped_labels(true, np.array([5, 5, -1, 7, 7, 7])) == pytest.approx(5 / 6)


# ── ICC (replaces the former GPL pingouin dependency) ───────────────────


def test_icc_matches_the_anova_definition():
    from scintilla.benchmarking.reproducibility import _compute_icc

    v = np.array([0.80, 0.82, 0.79, 0.81, 0.80, 0.83, 0.78, 0.82])
    a, b = v[:4], v[4:]
    ratings = np.column_stack([a, b])
    n, k = ratings.shape
    ms_b = k * np.sum((ratings.mean(axis=1) - ratings.mean()) ** 2) / (n - 1)
    ms_w = np.sum((ratings - ratings.mean(axis=1, keepdims=True)) ** 2) / (n * (k - 1))
    expected = np.clip((ms_b - ms_w) / (ms_b + (k - 1) * ms_w), 0, 1)
    assert _compute_icc(v) == pytest.approx(expected)
    assert _compute_icc(v, method="pingouin") == pytest.approx(expected)  # accepted alias


def test_icc_agrees_with_pingouin_when_it_is_installed():
    pingouin = pytest.importorskip("pingouin")
    from scintilla.benchmarking.reproducibility import _compute_icc

    v = np.array([0.80, 0.82, 0.79, 0.81, 0.80, 0.83, 0.78, 0.82])
    df = pd.DataFrame(
        {"t": list(range(4)) * 2, "r": ["A"] * 4 + ["B"] * 4, "y": v}
    )
    ref = pingouin.intraclass_corr(data=df, targets="t", raters="r", ratings="y")
    assert _compute_icc(v) == pytest.approx(float(ref.loc[ref.Type == "ICC1", "ICC"].iloc[0]), abs=1e-9)


def test_icc_of_identical_values_is_one():
    from scintilla.benchmarking.reproducibility import _compute_icc

    assert _compute_icc(np.full(8, 0.7)) == 1.0
