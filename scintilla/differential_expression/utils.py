"""DE utilities: volcano plot data preparation and filtering."""

from __future__ import annotations

import numpy as np
import pandas as pd


def volcano_plot_data(
    de_results: pd.DataFrame,
    log2fc_col: str = "log2fc",
    pval_col: str = "p_adjusted",
    log2fc_threshold: float = 1.0,
    alpha: float = 0.05,
) -> pd.DataFrame:
    """Annotate DE results with significance and regulation columns.

    Parameters
    ----------
    de_results:
        DE results DataFrame.
    log2fc_col:
        Column name for log2 fold change.
    pval_col:
        Column name for p-value.
    log2fc_threshold:
        Absolute log2FC threshold for significance.
    alpha:
        P-value threshold.

    Returns
    -------
    de_results with added 'significant' and 'regulation' columns.
    """
    df = de_results.copy()
    sig = (df[pval_col] < alpha) & (df[log2fc_col].abs() >= log2fc_threshold)
    df["significant"] = sig
    df["regulation"] = "not_significant"
    df.loc[sig & (df[log2fc_col] > 0), "regulation"] = "up"
    df.loc[sig & (df[log2fc_col] < 0), "regulation"] = "down"
    return df


def filter_de_genes(
    de_results: pd.DataFrame,
    log2fc_col: str = "log2fc",
    pval_col: str = "p_adjusted",
    log2fc_threshold: float = 1.0,
    alpha: float = 0.05,
    effect_size_col: str | None = None,
    effect_size_threshold: float = 0.3,
) -> pd.DataFrame:
    """Filter DE results to significant genes.

    Parameters
    ----------
    de_results:
        DE results DataFrame.
    log2fc_col:
        Column name for log2 fold change.
    pval_col:
        Column name for p-value.
    log2fc_threshold:
        Absolute log2FC threshold.
    alpha:
        P-value threshold.
    effect_size_col:
        Optional column name for an effect-size metric (e.g.
        ``"rank_biserial"``, ``"cohens_d"``).  When provided, an
        additional filter ``|effect_size| >= effect_size_threshold`` is
        applied.
    effect_size_threshold:
        Minimum absolute effect size.  Only used when *effect_size_col*
        is not None.

    Returns
    -------
    Filtered DataFrame.
    """
    mask = (de_results[pval_col] < alpha) & (de_results[log2fc_col].abs() >= log2fc_threshold)
    if effect_size_col is not None and effect_size_col in de_results.columns:
        mask = mask & (de_results[effect_size_col].abs() >= effect_size_threshold)
    return de_results[mask].reset_index(drop=True)
