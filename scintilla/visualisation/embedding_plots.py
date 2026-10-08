"""Embedding visualisation plots."""

from __future__ import annotations

from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np


def plot_embedding(
    adata,
    basis: str,
    colour_by: str,
    title: Optional[str] = None,
) -> plt.Figure:
    """Scatter plot of a 2D embedding coloured by a label.

    Parameters
    ----------
    adata:
        AnnData with embedding in obsm[basis].
    basis:
        Key in obsm (e.g., 'X_umap', 'X_tsne').
    colour_by:
        Column in obs to colour by.
    title:
        Plot title.

    Returns
    -------
    matplotlib Figure.
    """
    key = basis if basis.startswith("X_") else f"X_{basis}"
    if key not in adata.obsm:
        raise KeyError(f"Embedding '{key}' not found in obsm.")

    coords = adata.obsm[key][:, :2]
    labels = adata.obs[colour_by].values
    unique_labels = np.unique(labels)

    fig, ax = plt.subplots(figsize=(8, 6))
    for lbl in unique_labels:
        mask = labels == lbl
        ax.scatter(coords[mask, 0], coords[mask, 1], label=str(lbl), s=5, alpha=0.6)

    ax.set_xlabel(f"{key}1")
    ax.set_ylabel(f"{key}2")
    ax.set_title(title or f"{key} coloured by {colour_by}")
    ax.legend(markerscale=3, bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()
    return fig


def compare_embeddings(
    adata,
    bases: List[str],
    colour_by: str,
) -> plt.Figure:
    """Compare multiple embeddings side-by-side.

    Parameters
    ----------
    adata:
        AnnData.
    bases:
        List of obsm keys.
    colour_by:
        Column in obs to colour by.

    Returns
    -------
    matplotlib Figure with subplots.
    """
    n = len(bases)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 5))
    if n == 1:
        axes = [axes]

    labels = adata.obs[colour_by].values
    unique_labels = np.unique(labels)
    cmap = plt.cm.get_cmap("tab20", len(unique_labels))
    label_colour = {lbl: cmap(i) for i, lbl in enumerate(unique_labels)}

    for ax, basis in zip(axes, bases):
        key = basis if basis.startswith("X_") else f"X_{basis}"
        if key not in adata.obsm:
            ax.set_title(f"{key} (not found)")
            continue
        coords = adata.obsm[key][:, :2]
        colours = [label_colour[l] for l in labels]
        ax.scatter(coords[:, 0], coords[:, 1], c=colours, s=5, alpha=0.6)
        ax.set_title(key)
        ax.set_xlabel(f"{key}1")
        ax.set_ylabel(f"{key}2")

    plt.tight_layout()
    return fig
