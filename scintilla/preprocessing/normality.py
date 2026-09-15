"""Normality testing for single-cell expression data."""

from __future__ import annotations

import warnings
from typing import Optional, Tuple, Union

import anndata as ad
import numpy as np
import pandas as pd
from scipy import stats

from scintilla.config import RANDOM_SEED
from scintilla.io.loaders import ensure_anndata


def check_normality(
    data: Union[pd.DataFrame, ad.AnnData],
    alpha: float = 0.05,
    sample_size: int = 500,
    threshold: float = 0.3,
    n_replicates: int = 1,
    correction: Optional[str] = None,
    random_state: int = RANDOM_SEED,
) -> Tuple[bool, dict]:
    """Test normality of each feature using Shapiro-Wilk and Anderson-Darling.

    Parameters
    ----------
    data:
        Input data (DataFrame or AnnData).
    alpha:
        Significance level.
    sample_size:
        Maximum number of cells to subsample for Shapiro-Wilk test.
        Smaller values reduce the test's sensitivity to minor deviations
        from normality, which is appropriate for exploratory checks on
        noisy single-cell data (default 500).
    threshold:
        Fraction of features that must pass both tests for the overall
        result to be considered normal (default 0.3).  This is a
        *heuristic* tuning parameter, not a formal statistical cutoff.
        In typical scRNA-seq data, even well-normalised distributions
        rarely have > 20 % of genes passing Shapiro-Wilk at alpha = 0.05,
        so a strict 0.5 threshold almost always rejects normality.
    correction:
        Multiple-testing correction method applied to gene-wise Shapiro-Wilk
        p-values.  ``"bh"`` applies the Benjamini-Hochberg procedure and
        reports the fraction of genes significantly non-normal at
        FDR < *alpha*, replacing the arbitrary *threshold* with a formal
        multiple-testing framework.  ``None`` (default) uses the existing
        threshold-based approach.
    n_replicates:
        Number of independent subsampling rounds.  When > 1, the
        median pass fraction across replicates is used, which reduces
        sampling variability (default 1).

    Returns
    -------
    is_normal : bool
        True if the majority of features pass both tests.
    report : dict
        Per-feature statistics, p-values, pass/fail, and a summary string.
    """
    adata = ensure_anndata(data)
    X = adata.X if not hasattr(adata.X, "toarray") else adata.X.toarray()
    X = X.astype(np.float64)

    n_features = X.shape[1]
    anderson_stats = np.zeros(n_features)
    anderson_pass = np.zeros(n_features, dtype=bool)
    anderson_error = None

    # Anderson-Darling (full data, no subsampling needed)
    for j in range(n_features):
        col = X[:, j]
        ad_stat = np.nan
        ad_p = False
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", FutureWarning)
                ad_result = stats.anderson(col, dist="norm")
            ad_stat = ad_result.statistic
            ad_p = bool(ad_stat < ad_result.critical_values[2])
        except ValueError as exc:
            anderson_error = anderson_error or exc
        anderson_stats[j] = ad_stat
        anderson_pass[j] = ad_p

    if anderson_error is not None:
        warnings.warn(
            f"Anderson-Darling normality probe failed: {anderson_error}",
            stacklevel=2,
        )

    frac_anderson = float(np.mean(anderson_pass))

    # Shapiro-Wilk with optional replicate subsampling
    n_reps = max(1, n_replicates)
    replicate_fracs = []
    shapiro_pvals = np.zeros(n_features)
    shapiro_stats = np.zeros(n_features)
    shapiro_failed_indices = set()
    shapiro_failure_events = 0
    first_shapiro_error = None

    for rep in range(n_reps):
        rng = np.random.default_rng(random_state + rep)
        sw_pvals = np.full(n_features, np.nan)
        sw_stats = np.full(n_features, np.nan)
        for j in range(n_features):
            col = X[:, j]
            if len(col) > sample_size:
                col_sw = rng.choice(col, sample_size, replace=False)
            else:
                col_sw = col
            try:
                sw_stat, sw_p = stats.shapiro(col_sw)
            except (ValueError, TypeError, FloatingPointError) as exc:
                sw_stat, sw_p = np.nan, np.nan
                shapiro_failed_indices.add(j)
                shapiro_failure_events += 1
                first_shapiro_error = first_shapiro_error or exc
            sw_stats[j] = sw_stat
            sw_pvals[j] = sw_p

        sw_pass = np.isfinite(sw_pvals) & (sw_pvals > alpha)
        frac = float(np.mean(sw_pass))
        replicate_fracs.append(frac)
        if rep == 0:
            shapiro_pvals = sw_pvals
            shapiro_stats = sw_stats

    frac_shapiro = float(np.median(replicate_fracs))

    if shapiro_failed_indices:
        warnings.warn(
            "Shapiro-Wilk failed for "
            f"{len(shapiro_failed_indices)} of {n_features} features "
            f"({shapiro_failure_events} failure events): {first_shapiro_error}",
            UserWarning,
            stacklevel=2,
        )

    # Compute pass fractions at multiple alpha levels for richer reporting
    valid_pvals = shapiro_pvals[~np.isnan(shapiro_pvals)]
    pass_fractions_at_alphas = {}
    for a in (0.01, 0.05, 0.10):
        if len(valid_pvals) > 0:
            pass_fractions_at_alphas[str(a)] = float(np.mean(valid_pvals > a))
        else:
            pass_fractions_at_alphas[str(a)] = 0.0
    median_shapiro_pvalue = float(np.median(valid_pvals)) if len(valid_pvals) > 0 else float("nan")

    is_normal = (frac_shapiro > threshold) and (frac_anderson > threshold)

    report = {
        "shapiro_p_values": shapiro_pvals.tolist(),
        "shapiro_statistics": shapiro_stats.tolist(),
        "shapiro_pass_fraction": frac_shapiro,
        "anderson_statistics": anderson_stats.tolist(),
        "anderson_pass": anderson_pass.tolist(),
        "anderson_pass_fraction": frac_anderson,
        "median_shapiro_pvalue": median_shapiro_pvalue,
        "shapiro_pass_fractions_at_alphas": pass_fractions_at_alphas,
        "shapiro_failure_count": len(shapiro_failed_indices),
        "shapiro_failure_events": shapiro_failure_events,
        "shapiro_failed_features": [
            str(adata.var_names[j]) for j in sorted(shapiro_failed_indices)
        ],
        "alpha": alpha,
        "diagnostics": (
            f"Shapiro-Wilk: {frac_shapiro:.1%} features pass (alpha={alpha}). "
            f"Anderson-Darling: {frac_anderson:.1%} features pass. "
            f"Overall: {'NORMAL' if is_normal else 'NOT NORMAL'}."
        ),
    }

    # --- Benjamini-Hochberg corrected normality assessment ---
    if correction == "bh" and len(valid_pvals) > 0:
        from statsmodels.stats.multitest import multipletests  # noqa: PLC0415

        rejected, p_adj, _, _ = multipletests(valid_pvals, alpha=alpha, method="fdr_bh")
        bh_rejection_rate = float(np.mean(rejected))
        report["bh_adjusted_pvalues"] = p_adj.tolist()
        report["bh_rejection_rate"] = bh_rejection_rate
        report["n_genes_tested"] = len(valid_pvals)
        # Override the heuristic: data is "normal" if < 50% of genes are
        # rejected after BH correction (i.e. majority of genes are consistent
        # with normality at the controlled FDR level).
        is_normal = bh_rejection_rate < 0.5
        report["diagnostics"] = (
            f"BH-corrected normality: {bh_rejection_rate:.1%} of "
            f"{len(valid_pvals)} genes rejected at FDR < {alpha}. "
            f"Overall: {'NORMAL' if is_normal else 'NOT NORMAL'}."
        )

    return is_normal, report
