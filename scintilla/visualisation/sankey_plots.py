"""Sankey plot for cluster-label comparison."""

from __future__ import annotations

import numpy as np


def cluster_label_sankey(
    cluster_labels,
    true_labels,
    title: str = "Cluster-Label Sankey",
):
    """Create a Sankey diagram showing cluster to label flow.

    Parameters
    ----------
    cluster_labels:
        Predicted cluster labels.
    true_labels:
        True labels.
    title:
        Plot title.

    Returns
    -------
    plotly Figure.
    """
    try:
        import plotly.graph_objects as go  # noqa: PLC0415
    except ImportError as exc:
        raise ImportError(
            "plotly is required for Sankey plots. Install with: pip install plotly"
        ) from exc

    cluster_labels = np.asarray(cluster_labels)
    true_labels = np.asarray(true_labels)

    unique_clusters = sorted(np.unique(cluster_labels).tolist(), key=str)
    unique_true = sorted(np.unique(true_labels).tolist(), key=str)

    # Build source/target/value lists
    cluster_offset = 0
    true_offset = len(unique_clusters)

    cluster_idx = {c: i for i, c in enumerate(unique_clusters)}
    true_idx = {t: i + true_offset for i, t in enumerate(unique_true)}

    sources, targets, values = [], [], []
    for c in unique_clusters:
        for t in unique_true:
            count = int(((cluster_labels == c) & (true_labels == t)).sum())
            if count > 0:
                sources.append(cluster_idx[c])
                targets.append(true_idx[t])
                values.append(count)

    all_labels = [str(c) for c in unique_clusters] + [str(t) for t in unique_true]

    fig = go.Figure(data=[go.Sankey(
        node=dict(label=all_labels, pad=15, thickness=20),
        link=dict(source=sources, target=targets, value=values),
    )])
    fig.update_layout(title_text=title, font_size=10)
    return fig
