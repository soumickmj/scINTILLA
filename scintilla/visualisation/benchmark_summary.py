"""Benchmark summary visualisation."""

from __future__ import annotations

from typing import Dict, List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def radar_chart(
    metrics_dict: Dict[str, float],
    title: str = "Benchmark Radar Chart",
) -> plt.Figure:
    """Radar chart of multiple metrics for one method.

    Parameters
    ----------
    metrics_dict:
        Dictionary {metric_name: value}.
    title:
        Plot title.

    Returns
    -------
    matplotlib Figure.
    """
    labels = list(metrics_dict.keys())
    values = [float(v) for v in metrics_dict.values()]
    n = len(labels)

    angles = np.linspace(0, 2 * np.pi, n, endpoint=False).tolist()
    values_plot = values + [values[0]]
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw={"polar": True})
    ax.plot(angles, values_plot, "o-", linewidth=2)
    ax.fill(angles, values_plot, alpha=0.25)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels)
    ax.set_title(title)
    plt.tight_layout()
    return fig


def benchmark_heatmap(
    results_df: pd.DataFrame,
    method_col: str = "method",
    metric_cols: Optional[List[str]] = None,
    title: str = "Benchmark Summary",
) -> plt.Figure:
    """Heatmap of method x metric performance.

    Parameters
    ----------
    results_df:
        DataFrame with methods and metrics.
    method_col:
        Column name for methods.
    metric_cols:
        Columns to include as metrics. If None, uses all numeric columns.
    title:
        Plot title.

    Returns
    -------
    matplotlib Figure.
    """
    if metric_cols is None:
        metric_cols = [c for c in results_df.columns if c != method_col and
                       pd.api.types.is_numeric_dtype(results_df[c])]

    pivot = results_df.set_index(method_col)[metric_cols]

    fig, ax = plt.subplots(figsize=(max(6, len(metric_cols) * 1.5), max(4, len(pivot) * 0.6)))
    im = ax.imshow(pivot.values.astype(float), aspect="auto", cmap="RdYlGn")
    ax.set_xticks(range(len(metric_cols)))
    ax.set_xticklabels(metric_cols, rotation=45, ha="right")
    ax.set_yticks(range(len(pivot)))
    ax.set_yticklabels(pivot.index)
    plt.colorbar(im, ax=ax)
    ax.set_title(title)

    # Annotate cells
    for i in range(len(pivot)):
        for j in range(len(metric_cols)):
            val = pivot.values[i, j]
            if not np.isnan(float(val)):
                ax.text(j, i, f"{float(val):.2f}", ha="center", va="center", fontsize=8)

    plt.tight_layout()
    return fig
