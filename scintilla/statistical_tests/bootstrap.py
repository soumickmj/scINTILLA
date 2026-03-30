"""Bootstrap confidence intervals and related estimators.

Provides BCa (bias-corrected and accelerated) bootstrap, paired bootstrap
tests, the .632+ bootstrap estimator, and jackknife-after-bootstrap
diagnostics.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Optional, Tuple, Union

import numpy as np
from scipy import stats as sp_stats


# ---------------------------------------------------------------------------
# BCa bootstrap confidence interval
# ---------------------------------------------------------------------------

def bca_bootstrap_ci(
    data: np.ndarray,
    stat_fn: Callable[[np.ndarray], float],
    B: int = 2000,
    alpha: float = 0.05,
    seed: int = 42,
    method: str = "bca",
) -> Dict[str, float]:
    """Bias-corrected and accelerated (BCa) bootstrap confidence interval.

    Parameters
    ----------
    data:
        1-D or 2-D array.  Resampling is along axis 0.
    stat_fn:
        Function that takes an array (same shape as *data*) and returns a
        scalar statistic.
    B:
        Number of bootstrap replicates.
    alpha:
        Significance level (default 0.05 → 95 % CI).
    seed:
        Random seed for reproducibility.
    method:
        Bootstrap CI method.

        - ``"bca"`` (default) — full BCa with bias correction and
          acceleration via an O(n) jackknife loop.  Most accurate for
          small to moderate *n*.
        - ``"percentile"`` — simple percentile bootstrap.  Skips the
          jackknife entirely, making it much faster for large *n*
          (> 5 000).  Adequate when the statistic is approximately
          symmetric.

    Returns
    -------
    dict with keys ``point``, ``ci_low``, ``ci_high``.
    """
    import warnings  # noqa: PLC0415

    data = np.asarray(data)
    n = data.shape[0]
    rng = np.random.default_rng(seed)

    point = float(stat_fn(data))

    # --- bootstrap replicates ---
    boot_stats = np.empty(B, dtype=np.float64)
    for b in range(B):
        idx = rng.integers(0, n, size=n)
        boot_stats[b] = stat_fn(data[idx])

    # --- percentile fast-path (skip jackknife) ---
    if method == "percentile":
        lo = float(np.percentile(boot_stats, 100 * alpha / 2))
        hi = float(np.percentile(boot_stats, 100 * (1 - alpha / 2)))
        return {"point": point, "ci_low": lo, "ci_high": hi}

    # --- BCa: warn when n is large and jackknife will be slow ---
    if n > 5000:
        warnings.warn(
            f"bca_bootstrap_ci: n={n:,} is large — the O(n) jackknife "
            f"loop may be slow.  Consider method='percentile' for faster "
            f"(though slightly less accurate) confidence intervals.",
            stacklevel=2,
        )

    # --- bias correction (z0) ---
    z0 = sp_stats.norm.ppf(np.mean(boot_stats < point))
    if not np.isfinite(z0):
        # Fall back to percentile bootstrap when z0 is degenerate.
        lo = float(np.percentile(boot_stats, 100 * alpha / 2))
        hi = float(np.percentile(boot_stats, 100 * (1 - alpha / 2)))
        return {"point": point, "ci_low": lo, "ci_high": hi}

    # --- acceleration (a) via jackknife ---
    jack = np.empty(n, dtype=np.float64)
    for i in range(n):
        jack[i] = stat_fn(np.delete(data, i, axis=0))
    jack_mean = jack.mean()
    num = np.sum((jack_mean - jack) ** 3)
    den = 6.0 * (np.sum((jack_mean - jack) ** 2) ** 1.5)
    a = num / den if den != 0 else 0.0

    # --- adjusted percentiles ---
    z_alpha = sp_stats.norm.ppf(alpha / 2)
    z_1alpha = sp_stats.norm.ppf(1 - alpha / 2)

    def _adj(z: float) -> float:
        num_ = z0 + z
        return sp_stats.norm.cdf(z0 + num_ / (1 - a * num_))

    p_lo = max(0.0, min(1.0, _adj(z_alpha)))
    p_hi = max(0.0, min(1.0, _adj(z_1alpha)))

    lo = float(np.percentile(boot_stats, 100 * p_lo))
    hi = float(np.percentile(boot_stats, 100 * p_hi))

    return {"point": point, "ci_low": lo, "ci_high": hi}


# ---------------------------------------------------------------------------
# Convenience wrapper for classification metrics
# ---------------------------------------------------------------------------

def bootstrap_metric_ci(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metric_fn: Callable[[np.ndarray, np.ndarray], float],
    B: int = 2000,
    alpha: float = 0.05,
    seed: int = 42,
) -> Dict[str, float]:
    """Bootstrap CI for a metric computed on *(y_true, y_pred)* pairs.

    Parameters
    ----------
    y_true:
        Ground-truth labels.
    y_pred:
        Predicted labels (or probabilities).
    metric_fn:
        ``metric_fn(y_true, y_pred) -> float``.
    B:
        Bootstrap replicates.
    alpha:
        Significance level.
    seed:
        Random seed.

    Returns
    -------
    dict with ``point``, ``ci_low``, ``ci_high``.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    paired = np.column_stack([y_true, y_pred])

    def _stat(arr: np.ndarray) -> float:
        return metric_fn(arr[:, 0], arr[:, 1])

    return bca_bootstrap_ci(paired, _stat, B=B, alpha=alpha, seed=seed)


# ---------------------------------------------------------------------------
# Paired bootstrap test (two methods on the same test set)
# ---------------------------------------------------------------------------

def paired_bootstrap_test(
    y_true: np.ndarray,
    pred_a: np.ndarray,
    pred_b: np.ndarray,
    metric_fn: Callable[[np.ndarray, np.ndarray], float],
    B: int = 2000,
    seed: int = 42,
) -> Dict[str, float]:
    """Paired bootstrap test for comparing two sets of predictions.

    For each resample the metric is computed for both prediction vectors on
    the **same** resampled test set, and the distribution of differences is
    examined.

    Parameters
    ----------
    y_true, pred_a, pred_b:
        Ground-truth and predictions from methods A and B.
    metric_fn:
        ``metric_fn(y_true, y_pred) -> float``.
    B:
        Number of bootstrap replicates.
    seed:
        Random seed.

    Returns
    -------
    dict with ``diff`` (A − B point estimate), ``ci_low``, ``ci_high``,
    ``p_value`` (two-sided).
    """
    y_true = np.asarray(y_true)
    pred_a = np.asarray(pred_a)
    pred_b = np.asarray(pred_b)
    n = len(y_true)
    rng = np.random.default_rng(seed)

    obs_diff = metric_fn(y_true, pred_a) - metric_fn(y_true, pred_b)

    diffs = np.empty(B, dtype=np.float64)
    for b in range(B):
        idx = rng.integers(0, n, size=n)
        diffs[b] = metric_fn(y_true[idx], pred_a[idx]) - metric_fn(y_true[idx], pred_b[idx])

    ci_low = float(np.percentile(diffs, 2.5))
    ci_high = float(np.percentile(diffs, 97.5))
    # Two-sided p-value: fraction of bootstrap diffs on the opposite side of zero
    p_value = float(np.mean(np.abs(diffs - np.mean(diffs)) >= abs(obs_diff)))

    return {
        "diff": float(obs_diff),
        "ci_low": ci_low,
        "ci_high": ci_high,
        "p_value": p_value,
    }


# ---------------------------------------------------------------------------
# .632+ bootstrap estimator
# ---------------------------------------------------------------------------

def _analytical_no_info_rate(y: np.ndarray) -> float:
    """Analytical no-information rate for accuracy: sum of squared class proportions.

    Under independence of true labels and predictions, the expected
    accuracy equals ``sum_k(p_k * q_k)`` where ``p_k`` and ``q_k`` are the
    true and predicted class proportions respectively.  Since under the
    null the predicted distribution matches the true distribution,
    ``gamma = sum_k(p_k^2)``.

    This is exact and has zero variance, unlike a permutation-based estimate.
    """
    _, counts = np.unique(y, return_counts=True)
    proportions = counts / len(y)
    return float(np.sum(proportions ** 2))


def dot632plus_bootstrap(
    model_cls: Any,
    X: np.ndarray,
    y: np.ndarray,
    metric_fn: Callable[[np.ndarray, np.ndarray], float],
    B: int = 200,
    seed: int = 42,
    no_info_method: str = "analytical",
    n_permutations: int = 50,
) -> Dict[str, float]:
    """.632+ bootstrap error estimator (Efron & Tibshirani 1997).

    Parameters
    ----------
    model_cls:
        An *unfitted* scikit-learn-compatible estimator (has ``fit`` /
        ``predict``).  A fresh clone is created for each bootstrap iteration.
    X:
        Feature matrix.
    y:
        Labels.
    metric_fn:
        ``metric_fn(y_true, y_pred) -> float``.  Higher is better.
    B:
        Bootstrap iterations.
    seed:
        Random seed.
    no_info_method:
        How to estimate the no-information rate (gamma).

        - ``"analytical"`` (default) — ``sum(p_k^2)`` where ``p_k`` are
          the class proportions.  Exact, zero-variance, and correct for
          accuracy.  For other metrics it is still a reasonable proxy.
        - ``"permutation"`` — average the metric over *n_permutations*
          random label permutations (the original approach used a single
          permutation).
    n_permutations:
        Number of permutations when ``no_info_method="permutation"``.

    Returns
    -------
    dict with ``estimate``, ``ci_low``, ``ci_high``.
    """
    from sklearn.base import clone  # noqa: PLC0415

    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y)
    n = len(y)
    rng = np.random.default_rng(seed)

    # No-information rate
    if no_info_method == "analytical":
        gamma = _analytical_no_info_rate(y)
        # The analytical formula sum(p_k^2) is exact only for accuracy.
        # For other metrics (e.g. macro-F1) it is an approximation that
        # can be substantially wrong on imbalanced datasets.
        try:
            from sklearn.metrics import accuracy_score as _acc  # noqa: PLC0415
            _is_accuracy = (metric_fn is _acc
                            or getattr(metric_fn, "__name__", "") == "accuracy_score")
        except Exception:
            _is_accuracy = False
        if not _is_accuracy:
            import warnings as _w  # noqa: PLC0415
            _w.warn(
                "dot632plus_bootstrap: analytical no-information rate "
                "(sum of squared class proportions) is exact only for "
                "accuracy.  For other metrics consider using "
                "no_info_method='permutation' for an unbiased estimate.",
                stacklevel=2,
            )
    else:
        # Average over multiple permutations for a stable estimate
        gamma_vals = np.empty(n_permutations, dtype=np.float64)
        for p in range(n_permutations):
            gamma_vals[p] = metric_fn(y, rng.permutation(y))
        gamma = float(np.mean(gamma_vals))

    estimates = np.empty(B, dtype=np.float64)
    for b in range(B):
        idx = rng.integers(0, n, size=n)
        oob_mask = np.ones(n, dtype=bool)
        oob_mask[idx] = False
        if not oob_mask.any():
            estimates[b] = np.nan
            continue

        model = clone(model_cls)
        model.fit(X[idx], y[idx])

        # Training error
        train_pred = model.predict(X[idx])
        train_err = metric_fn(y[idx], train_pred)

        # OOB error
        oob_pred = model.predict(X[oob_mask])
        oob_err = metric_fn(y[oob_mask], oob_pred)

        # .632+ weight
        r = (oob_err - train_err) / (gamma - train_err + 1e-10)
        r = np.clip(r, 0, 1)
        w = 0.632 / (1 - 0.368 * r)
        w = np.clip(w, 0.632, 1.0)

        estimates[b] = (1 - w) * train_err + w * oob_err

    valid = estimates[np.isfinite(estimates)]
    point = float(np.mean(valid)) if len(valid) > 0 else float("nan")
    ci_low = float(np.percentile(valid, 2.5)) if len(valid) > 0 else float("nan")
    ci_high = float(np.percentile(valid, 97.5)) if len(valid) > 0 else float("nan")

    return {"estimate": point, "ci_low": ci_low, "ci_high": ci_high}


# ---------------------------------------------------------------------------
# Jackknife-after-bootstrap
# ---------------------------------------------------------------------------

def jackknife_after_bootstrap(
    data: np.ndarray,
    stat_fn: Callable[[np.ndarray], float],
    B: int = 2000,
    seed: int = 42,
) -> Dict[str, Any]:
    """Jackknife-after-bootstrap to identify influential observations.

    Parameters
    ----------
    data:
        Input array (axis 0 = observations).
    stat_fn:
        Statistic function.
    B:
        Bootstrap replicates.
    seed:
        Random seed.

    Returns
    -------
    dict with ``influence_scores`` (array, one per observation) and
    ``influential_indices`` (those exceeding 2× IQR above Q3).
    """
    data = np.asarray(data)
    n = data.shape[0]
    rng = np.random.default_rng(seed)

    # Generate bootstrap indices
    boot_indices = rng.integers(0, n, size=(B, n))

    # Full bootstrap distribution
    full_stats = np.array([stat_fn(data[boot_indices[b]]) for b in range(B)])

    # Leave-one-out bootstrap distributions
    influence = np.empty(n, dtype=np.float64)
    for i in range(n):
        # Keep only replicates that did NOT draw observation i
        mask = np.array([i not in boot_indices[b] for b in range(B)])
        if mask.sum() < 10:
            influence[i] = 0.0
            continue
        loo_mean = full_stats[mask].mean()
        influence[i] = abs(full_stats.mean() - loo_mean)

    q1, q3 = np.percentile(influence, [25, 75])
    iqr = q3 - q1
    threshold = q3 + 2.0 * iqr
    influential = np.where(influence > threshold)[0]

    return {
        "influence_scores": influence,
        "influential_indices": influential,
    }


# ---------------------------------------------------------------------------
# Simple bootstrap for a 1-D vector of metric values
# ---------------------------------------------------------------------------

def bootstrap_resample_metrics(
    values: np.ndarray,
    B: int = 2000,
    alpha: float = 0.05,
    seed: int = 42,
) -> Dict[str, float]:
    """Bootstrap a 1-D array of metric values (e.g. per-seed scores).

    Parameters
    ----------
    values:
        1-D array of metric values.
    B:
        Bootstrap replicates.
    alpha:
        Significance level.
    seed:
        Random seed.

    Returns
    -------
    dict with ``mean``, ``ci_low``, ``ci_high``.
    """
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return {"mean": float("nan"), "ci_low": float("nan"), "ci_high": float("nan")}

    rng = np.random.default_rng(seed)
    means = np.empty(B, dtype=np.float64)
    n = len(values)
    for b in range(B):
        means[b] = values[rng.integers(0, n, size=n)].mean()

    return {
        "mean": float(values.mean()),
        "ci_low": float(np.percentile(means, 100 * alpha / 2)),
        "ci_high": float(np.percentile(means, 100 * (1 - alpha / 2))),
    }
