"""Clustering visualisation utilities."""

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


def plot_clustering_benchmark(results_df: pd.DataFrame, axes=None) -> plt.Figure:
    """Two-panel bar chart (ARI and AMI) of a clustering benchmark.

    Parameters
    ----------
    results_df
        The table returned by :func:`scintilla.clustering.benchmark.benchmark_clustering_methods`.
    axes
        Optional pair of matplotlib axes ``(ax_ari, ax_ami)``; a new figure is created
        when omitted.

    Returns
    -------
    matplotlib.figure.Figure
        The figure; it is not closed.
    """
    valid_df = results_df.dropna(subset=["ari"]).sort_values("ari", ascending=True)
    if axes is None:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, max(4, len(results_df) * 0.25)))
    else:
        ax1, ax2 = axes
        fig = ax1.figure
    if not valid_df.empty:
        y_labels = valid_df["method"] + " | " + valid_df["params"].astype(str)
        y_pos = range(len(valid_df))
        ax1.barh(y_pos, valid_df["ari"].values, color="steelblue")
        ax1.set_yticks(y_pos)
        ax1.set_yticklabels(y_labels, fontsize=6)
        ax1.set_xlabel("ARI")
        ax1.set_title("Clustering Benchmark (ARI)")

        ami_sorted = valid_df.sort_values("ami", ascending=True)
        y_labels_ami = ami_sorted["method"] + " | " + ami_sorted["params"].astype(str)
        y_pos_ami = range(len(ami_sorted))
        ax2.barh(y_pos_ami, ami_sorted["ami"].values, color="darkorange")
        ax2.set_yticks(y_pos_ami)
        ax2.set_yticklabels(y_labels_ami, fontsize=6)
        ax2.set_xlabel("AMI")
        ax2.set_title("Clustering Benchmark (AMI)")
    fig.tight_layout()
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
