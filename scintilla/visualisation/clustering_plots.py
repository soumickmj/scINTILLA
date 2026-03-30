"""Clustering visualisation utilities."""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import dendrogram as _dendrogram


def ari_benchmark_plot(
    results_df: pd.DataFrame,
    title: str = "Clustering Benchmark",
) -> plt.Figure:
    """Horizontal bar chart of ARI scores per method."""
    valid = results_df.dropna(subset=["ari"]).sort_values("ari", ascending=True)
    labels = valid["method"] + " | " + valid["params"].astype(str)
    fig, ax = plt.subplots(figsize=(12, max(4, len(valid) * 0.3)))
    ax.barh(range(len(valid)), valid["ari"].values, color="steelblue")
    ax.set_yticks(range(len(valid)))
    ax.set_yticklabels(labels, fontsize=7)
    ax.set_xlabel("ARI")
    ax.set_title(title)
    plt.tight_layout()
    return fig


def dendrogram_plot(
    Z: np.ndarray,
    labels=None,
    title: str = "Dendrogram",
) -> plt.Figure:
    """Plot a scipy linkage matrix as a dendrogram."""
    fig, ax = plt.subplots(figsize=(12, 5))
    _dendrogram(Z, ax=ax, labels=labels, truncate_mode="lastp", p=30, no_labels=(labels is None))
    ax.set_title(title)
    plt.tight_layout()
    return fig


def cophenetic_vs_original_scatter(
    cophenetic_distances: np.ndarray,
    original_distances: np.ndarray,
    title: str = "Cophenetic vs Original Distances",
) -> plt.Figure:
    """Scatter plot of cophenetic vs original distances."""
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(original_distances, cophenetic_distances, alpha=0.3, s=5)
    ax.set_xlabel("Original distances")
    ax.set_ylabel("Cophenetic distances")
    ax.set_title(title)
    plt.tight_layout()
    return fig
