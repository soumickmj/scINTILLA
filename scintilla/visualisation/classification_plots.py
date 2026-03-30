"""Classification visualisation utilities."""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def benchmark_bar_chart(
    results_df: pd.DataFrame,
    metric: str = "accuracy",
    title: str = "Classification Benchmark",
) -> plt.Figure:
    """Grouped bar chart of classification metric per model."""
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
    """Heatmap of a confusion matrix."""
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
