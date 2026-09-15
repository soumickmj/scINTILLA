"""Differential expression visualisation plots."""

from __future__ import annotations

from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def volcano_plot(
    de_results: pd.DataFrame,
    log2fc_col: str = "log2fc",
    pval_col: str = "p_adjusted",
    log2fc_threshold: float = 1.0,
    alpha: float = 0.05,
    title: str = "Volcano Plot",
) -> plt.Figure:
    """Volcano plot of DE results.

    Parameters
    ----------
    de_results:
        DE results DataFrame.
    log2fc_col:
        Column with log2 fold change.
    pval_col:
        Column with p-values.
    log2fc_threshold:
        Fold change threshold for significance.
    alpha:
        P-value threshold.
    title:
        Plot title.

    Returns
    -------
    matplotlib Figure.
    """
    df = de_results.copy()
    neg_log10_p = -np.log10(df[pval_col].values + 1e-300)
    sig = (df[pval_col] < alpha) & (df[log2fc_col].abs() >= log2fc_threshold)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(df[log2fc_col][~sig], neg_log10_p[~sig], c="grey", s=5, alpha=0.4, label="Not significant")
    ax.scatter(df[log2fc_col][sig], neg_log10_p[sig], c="red", s=10, alpha=0.8, label="Significant")
    ax.axhline(-np.log10(alpha), color="blue", linestyle="--", linewidth=0.8)
    ax.axvline(log2fc_threshold, color="green", linestyle="--", linewidth=0.8)
    ax.axvline(-log2fc_threshold, color="green", linestyle="--", linewidth=0.8)
    ax.set_xlabel("log2 Fold Change")
    ax.set_ylabel("-log10(p-value)")
    ax.set_title(title)
    ax.legend()
    plt.tight_layout()
    return fig


def ma_plot(
    de_results: pd.DataFrame,
    log2fc_col: str = "log2fc",
    mean_expr_col: Optional[str] = None,
    title: str = "MA Plot",
) -> plt.Figure:
    """MA plot (mean expression vs log2FC).

    Parameters
    ----------
    de_results:
        DE results DataFrame.
    log2fc_col:
        Column with log2 fold change.
    mean_expr_col:
        Column with mean expression. If None, uses sequential index.
    title:
        Plot title.

    Returns
    -------
    matplotlib Figure.
    """
    df = de_results.copy()
    if mean_expr_col is not None and mean_expr_col in df.columns:
        A = df[mean_expr_col].values
    else:
        A = np.arange(len(df))

    M = df[log2fc_col].values

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(A, M, s=5, alpha=0.4, c="steelblue")
    ax.axhline(0, color="red", linestyle="--", linewidth=1.0)
    ax.set_xlabel("Mean Expression (A)")
    ax.set_ylabel("log2 Fold Change (M)")
    ax.set_title(title)
    plt.tight_layout()
    return fig
