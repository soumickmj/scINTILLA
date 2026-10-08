"""Classification visualisation utilities."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def plot_classification_benchmark(results_df: pd.DataFrame, ax=None, title: str = None) -> plt.Figure:
    """Draw a grouped bar chart of accuracy per model and feature space.

    Handles the three table layouts that
    :func:`scintilla.classification.benchmark.benchmark_models_comprehensive` produces:
    cross-validation (``mean_accuracy`` with standard-error bars), hold-out and
    .632+ bootstrap (plain ``accuracy``).

    Parameters
    ----------
    results_df
        The benchmark table.
    ax
        Optional matplotlib axes to draw on; a new figure is created when omitted.
    title
        Optional title; a descriptive one is generated when omitted.

    Returns
    -------
    matplotlib.figure.Figure
        The figure; it is not closed.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 5))
    else:
        fig = ax.figure
    if "mean_accuracy" in results_df.columns and results_df["mean_accuracy"].notna().any():
        pivot = results_df.pivot_table(index="model", columns="space", values="mean_accuracy")
        # Error bars are the standard error of the mean (SD / sqrt(k)), the uncertainty
        # of the mean estimate, not the between-fold variability.
        if "se_accuracy" in results_df.columns:
            yerr = results_df.pivot_table(index="model", columns="space", values="se_accuracy")
            label = "± SE"
        else:
            yerr = results_df.pivot_table(index="model", columns="space", values="std_accuracy")
            label = "± SD"
        pivot.plot(kind="bar", ax=ax, yerr=yerr)
        ax.set_title(title or f"Classification Benchmark (cross-validation, mean accuracy {label})")
    elif "accuracy" in results_df.columns and results_df["accuracy"].notna().any():
        pivot = results_df.pivot_table(index="model", columns="space", values="accuracy")
        pivot.plot(kind="bar", ax=ax)
        ax.set_title(title or "Classification Benchmark (accuracy)")
    ax.set_ylabel("Accuracy")
    ax.set_xlabel("Model")
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    return fig


def benchmark_bar_chart(
    results_df: pd.DataFrame,
    metric: str = "accuracy",
    title: str = "Classification Benchmark",
) -> plt.Figure:
    """Draw a grouped bar chart of a classification metric per model.

    Parameters
    ----------
    results_df
        Benchmark results table.
    metric
        Column of ``results_df`` to plot.
    title
        Title of the plot.
    """
    fig, ax = plt.subplots(figsize=(10, 5))
    if metric in results_df.columns and "model" in results_df.columns:
        if "space" in results_df.columns:
            pivot = results_df.pivot_table(index="model", columns="space", values=metric)
            pivot.plot(kind="bar", ax=ax)
        else:
            results_df.set_index("model")[metric].plot(kind="bar", ax=ax)
    ax.set_title(title)
    ax.set_ylabel(metric.capitalize())
    ax.set_xlabel("Model")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    return fig


def confusion_matrix_heatmap(
    cm: np.ndarray,
    class_names=None,
    title: str = "Confusion Matrix",
) -> plt.Figure:
    """Heatmap of a confusion matrix.

    Parameters
    ----------
    cm
        Confusion matrix.
    class_names
        Class names for the axes.
    title
        Title of the plot.
    """
    fig, ax = plt.subplots(figsize=(max(4, cm.shape[0]), max(4, cm.shape[0])))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names or "auto",
        yticklabels=class_names or "auto",
        ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    plt.tight_layout()
    return fig
