"""Reproducibility / seed stability testing."""

from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd

from scintilla.benchmarking.profiler import profile_method


def seed_stability_test(
    method_fn: Callable,
    data,
    metric_fn: Callable,
    n_seeds: int = 10,
    base_seed: int = 42,
    bootstrap_ci: bool = False,
    n_bootstrap: int = 2000,
    compute_icc: bool = False,
    icc_method: str = "pingouin",
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

        - ``"pingouin"`` (default) — uses ``pingouin.intraclass_corr``
          with a split-half design for statistically correct ICC(1,1).
        - ``"legacy"`` — original formula (kept for backwards
          compatibility; note that with a single observation per seed
          this value is *not* a proper ICC).

    Returns
    -------
    pd.DataFrame  columns=[seed, metric_value]
    with added attribute stability_score (1 - cv).
    """
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
        from scintilla.statistical_tests.bootstrap import bootstrap_resample_metrics  # noqa: PLC0415
        ci = bootstrap_resample_metrics(values, B=n_bootstrap)
        df.attrs["ci_low"] = ci["ci_low"]
        df.attrs["ci_high"] = ci["ci_high"]

    # ICC(1,1)
    if compute_icc and len(values) >= 3:
        df.attrs["icc"] = _compute_icc(values, method=icc_method)

    return df


def _compute_icc(values: np.ndarray, method: str = "pingouin") -> float:
    """Compute ICC(1,1) from a vector of per-seed metric scores.

    Parameters
    ----------
    values:
        1-D array of metric values (one per seed, NaNs already removed).
    method:
        ``"pingouin"`` for a proper ICC via split-half reliability, or
        ``"legacy"`` for the original (approximate) formula.
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

    # --- pingouin-based ICC(1,1) via split-half design ---
    # With one observation per seed, we construct a two-rater design by
    # splitting the seed vector into two interleaved halves and computing
    # ICC(1,1) across the halves.  This gives a proper reliability
    # estimate for the method's consistency across seeds.
    try:
        import pingouin as pg  # noqa: PLC0415

        # Build long-form DataFrame: targets = pair index, raters = half
        n_pairs = k // 2
        if n_pairs < 2:
            # Not enough seeds for a split-half — fall back to legacy
            return _compute_icc(values, method="legacy")

        half_a = values[:n_pairs]
        half_b = values[n_pairs : 2 * n_pairs]

        icc_df = pd.DataFrame({
            "targets": list(range(n_pairs)) * 2,
            "raters": ["A"] * n_pairs + ["B"] * n_pairs,
            "ratings": np.concatenate([half_a, half_b]),
        })
        icc_table = pg.intraclass_corr(
            data=icc_df, targets="targets", raters="raters", ratings="ratings",
        )
        # ICC1 is the first row (Type ICC1)
        icc_val = float(icc_table.loc[icc_table["Type"] == "ICC1", "ICC"].iloc[0])
        return float(np.clip(icc_val, 0.0, 1.0))
    except Exception:
        # If pingouin is unavailable or computation fails, fall back
        return _compute_icc(values, method="legacy")
