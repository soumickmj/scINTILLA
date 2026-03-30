"""Permutation tests and McNemar's test for pairwise method comparison."""

from __future__ import annotations

from typing import Callable, Dict

import numpy as np
from scipy import stats as sp_stats


def permutation_test_methods(
    y_true: np.ndarray,
    pred_a: np.ndarray,
    pred_b: np.ndarray,
    metric_fn: Callable[[np.ndarray, np.ndarray], float],
    n_permutations: int = 1000,
    seed: int = 42,
) -> Dict[str, float]:
    """Permutation test for comparing two prediction vectors.

    For each permutation the method labels (A vs B) are randomly swapped for
    each observation and the metric difference is recomputed under the null
    hypothesis that both methods are equivalent.

    Parameters
    ----------
    y_true:
        Ground-truth labels.
    pred_a, pred_b:
        Predictions from methods A and B.
    metric_fn:
        ``metric_fn(y_true, y_pred) -> float``.
    n_permutations:
        Number of random permutations.
    seed:
        Random seed.

    Returns
    -------
    dict with ``observed_diff`` (A − B), ``p_value`` (two-sided).
    """
    y_true = np.asarray(y_true)
    pred_a = np.asarray(pred_a)
    pred_b = np.asarray(pred_b)
    n = len(y_true)
    rng = np.random.default_rng(seed)

    obs_diff = metric_fn(y_true, pred_a) - metric_fn(y_true, pred_b)

    count = 0
    for _ in range(n_permutations):
        swap = rng.random(n) < 0.5
        perm_a = np.where(swap, pred_b, pred_a)
        perm_b = np.where(swap, pred_a, pred_b)
        perm_diff = metric_fn(y_true, perm_a) - metric_fn(y_true, perm_b)
        if abs(perm_diff) >= abs(obs_diff):
            count += 1

    p_value = (count + 1) / (n_permutations + 1)  # +1 for continuity correction

    return {"observed_diff": float(obs_diff), "p_value": float(p_value)}


def mcnemar_test(
    y_true: np.ndarray,
    pred_a: np.ndarray,
    pred_b: np.ndarray,
) -> Dict[str, float]:
    """McNemar's test for paired nominal predictions.

    Tests whether two classifiers make the same pattern of errors.  Uses the
    exact binomial test when the number of discordant pairs is small (< 25),
    otherwise uses the chi-squared approximation with Edwards' continuity
    correction.

    Parameters
    ----------
    y_true:
        Ground-truth labels.
    pred_a, pred_b:
        Predictions from classifiers A and B.

    Returns
    -------
    dict with ``statistic``, ``p_value``, ``n_discordant``.
    """
    y_true = np.asarray(y_true)
    pred_a = np.asarray(pred_a)
    pred_b = np.asarray(pred_b)

    correct_a = pred_a == y_true
    correct_b = pred_b == y_true

    # b: A correct, B wrong; c: A wrong, B correct
    b = int(np.sum(correct_a & ~correct_b))
    c = int(np.sum(~correct_a & correct_b))
    n_discordant = b + c

    if n_discordant == 0:
        return {"statistic": 0.0, "p_value": 1.0, "n_discordant": 0}

    if n_discordant < 25:
        # Exact binomial test
        p_value = float(sp_stats.binomtest(b, n_discordant, 0.5).pvalue)
    else:
        # Chi-squared with Edwards' continuity correction
        stat = (abs(b - c) - 1) ** 2 / (b + c)
        p_value = float(1 - sp_stats.chi2.cdf(stat, df=1))

    chi2_stat = (abs(b - c) - 1) ** 2 / (b + c) if n_discordant > 0 else 0.0

    return {
        "statistic": float(chi2_stat),
        "p_value": float(p_value),
        "n_discordant": n_discordant,
    }
