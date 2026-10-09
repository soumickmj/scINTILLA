"""Reproducibility / seed stability testing."""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable, Optional

import numpy as np
import pandas as pd

from scintilla.config import RANDOM_SEED

if TYPE_CHECKING:
    from scintilla.analysis_config import AnalysisConfig


def seed_stability_test(
    method_fn: Callable,
    data,
    metric_fn: Callable,
    n_seeds: int = 10,
    base_seed: Optional[int] = None,
    bootstrap_ci: Optional[bool] = None,
    n_bootstrap: Optional[int] = None,
    compute_icc: bool = False,
    icc_method: str = "anova",
    config: Optional[AnalysisConfig] = None,
) -> pd.DataFrame:
    """Test seed stability of a method.

    Runs method_fn with n_seeds different random seeds and evaluates
    metric_fn on each result.

    Parameters
    ----------
    method_fn:
        Function called as method_fn(data, random_state=seed).
    data:
        Data passed to method_fn.
    metric_fn:
        Function called as metric_fn(result) returning a float metric.
    n_seeds:
        Number of seeds.
    base_seed:
        Starting seed.
    bootstrap_ci:
        If True, compute BCa bootstrap CIs for the mean metric over
        seeds.  Stored in ``df.attrs["ci_low"]`` / ``df.attrs["ci_high"]``.
    n_bootstrap:
        Bootstrap replicates for CIs.
    compute_icc:
        If True, compute ICC(1,1) to quantify the fraction of variance
        attributable to systematic method differences vs. seed noise.
        Stored in ``df.attrs["icc"]``.
    icc_method:
        How to compute the ICC.

        - ``"anova"`` (default) — one-way random-effects ICC(1,1) on a
          split-half design.  ``"pingouin"`` is accepted as an alias for
          code written against 0.1.0; it gives the same value and no longer
          needs the ``pingouin`` package.
        - ``"legacy"`` — original formula (kept for backwards
          compatibility; note that with a single observation per seed
          this value is *not* a proper ICC).

    Returns
    -------
    pd.DataFrame  columns=[seed, metric_value]
    with added attribute stability_score (1 - cv).
    """
    if base_seed is None:
        base_seed = getattr(config, "random_seed", RANDOM_SEED) if config is not None else RANDOM_SEED
    if bootstrap_ci is None:
        bootstrap_ci = getattr(config, "bootstrap_ci", False) if config is not None else False
    if n_bootstrap is None:
        n_bootstrap = getattr(config, "n_bootstrap", 2000) if config is not None else 2000

    records = []
    for i in range(n_seeds):
        seed = base_seed + i
        try:
            result = method_fn(data, random_state=seed)
            metric_val = metric_fn(result)
        except Exception:
            metric_val = float("nan")
        records.append({"seed": seed, "metric_value": metric_val})

    df = pd.DataFrame(records)
    values = df["metric_value"].dropna().values
    if len(values) > 0 and values.mean() != 0:
        cv = values.std() / (abs(values.mean()) + 1e-10)
        df.attrs["stability_score"] = float(1.0 - cv)
    else:
        df.attrs["stability_score"] = float("nan")

    # Bootstrap CIs on the per-seed metric vector
    if bootstrap_ci and len(values) >= 2:
        from scintilla.statistical_tests.bootstrap import bootstrap_resample_metrics
        ci = bootstrap_resample_metrics(values, B=n_bootstrap, seed=base_seed)
        df.attrs["ci_low"] = ci["ci_low"]
        df.attrs["ci_high"] = ci["ci_high"]

    # ICC(1,1)
    if compute_icc and len(values) >= 3:
        df.attrs["icc"] = _compute_icc(values, method=icc_method)

    return df


def _compute_icc(values: np.ndarray, method: str = "anova") -> float:
    """Compute ICC(1,1) from a vector of per-seed metric scores.

    Parameters
    ----------
    values:
        1-D array of metric values (one per seed, NaNs already removed).
    method:
        ``"anova"`` (alias ``"pingouin"``) for a proper ICC via split-half
        reliability, or ``"legacy"`` for the original (approximate) formula.
    """
    values = np.asarray(values, dtype=np.float64)
    k = len(values)

    # Perfect agreement — all values identical → ICC = 1.0
    if np.ptp(values) == 0:
        return 1.0

    if method == "legacy":
        grand_mean = float(np.mean(values))
        ms_between = float(np.sum((values - grand_mean) ** 2) / (k - 1))
        ms_within = float(np.var(values, ddof=1))
        icc = (ms_between - ms_within) / (ms_between + (k - 1) * ms_within + 1e-10)
        return float(np.clip(icc, 0, 1))

    # --- ICC(1,1) via a split-half design ---
    # With one observation per seed, we construct a two-rater design by
    # splitting the seed vector into two halves and computing the one-way
    # random-effects ICC(1,1) across the halves.  This is the classical
    # ANOVA estimator (Shrout & Fleiss, 1979), computed directly so that no
    # GPL-licensed statistics package is needed.
    n_pairs = k // 2
    if n_pairs < 2:
        # Not enough seeds for a split-half, fall back to the legacy estimator.
        return _compute_icc(values, method="legacy")

    ratings = np.column_stack([values[:n_pairs], values[n_pairs : 2 * n_pairs]])
    icc_val = _icc1(ratings)
    if not np.isfinite(icc_val):
        return _compute_icc(values, method="legacy")
    return float(np.clip(icc_val, 0.0, 1.0))


def _icc1(ratings: np.ndarray) -> float:
    """One-way random-effects ICC(1,1) of an ``(n_targets, k_raters)`` matrix."""
    n, k = ratings.shape
    grand = ratings.mean()
    target_means = ratings.mean(axis=1)
    ms_between = k * np.sum((target_means - grand) ** 2) / (n - 1)
    ms_within = np.sum((ratings - target_means[:, None]) ** 2) / (n * (k - 1))
    denom = ms_between + (k - 1) * ms_within
    if denom == 0:
        return float("nan")
    return float((ms_between - ms_within) / denom)
