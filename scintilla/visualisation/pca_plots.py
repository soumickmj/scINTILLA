"""PCA visualisation utilities."""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import anndata as ad


def pca_2d_plot(
    adata: ad.AnnData,
    colour_col=None,
    title: str = "PCA 2D",
) -> plt.Figure:
    """2D scatter plot of the first two PCs."""
    if "X_pca" not in adata.obsm:
        raise ValueError("Run PCA first (X_pca not in obsm).")
    X = adata.obsm["X_pca"]
    fig, ax = plt.subplots(figsize=(7, 6))
    if colour_col and colour_col in adata.obs.columns:
        cats = adata.obs[colour_col].values
        unique = np.unique(cats)
        cmap = plt.cm.get_cmap("tab20", len(unique))
        for i, cat in enumerate(unique):
            mask = cats == cat
            ax.scatter(X[mask, 0], X[mask, 1], label=cat, s=10, color=cmap(i), alpha=0.7)
        ax.legend(fontsize=7, markerscale=2, bbox_to_anchor=(1.05, 1))
    else:
        ax.scatter(X[:, 0], X[:, 1], s=10, alpha=0.7)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title(title)
    plt.tight_layout()
    return fig


def pca_3d_multiview(
    adata: ad.AnnData,
    colour_col=None,
) -> plt.Figure:
    """Three viewing angles of PC1-2-3 space."""
    if "X_pca" not in adata.obsm:
        raise ValueError("Run PCA first (X_pca not in obsm).")
    X = adata.obsm["X_pca"]
    angles = [(30, 30), (10, 90), (60, 120)]
    fig = plt.figure(figsize=(15, 5))
    cats = adata.obs[colour_col].values if colour_col and colour_col in adata.obs.columns else None
    unique = np.unique(cats) if cats is not None else None
    cmap = plt.cm.get_cmap("tab20", len(unique)) if unique is not None else None
    for k, (elev, azim) in enumerate(angles):
        ax = fig.add_subplot(1, 3, k + 1, projection="3d")
        if cats is not None and unique is not None:
            for i, cat in enumerate(unique):
                mask = cats == cat
                ax.scatter(X[mask, 0], X[mask, 1], X[mask, 2], label=cat, s=5, color=cmap(i), alpha=0.6)
        else:
            ax.scatter(X[:, 0], X[:, 1], X[:, 2], s=5, alpha=0.6)
        ax.view_init(elev=elev, azim=azim)
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.set_zlabel("PC3")
    plt.tight_layout()
    return fig


def cumulative_variance_plot(
    adata: ad.AnnData,
    threshold=None,
) -> plt.Figure:
    """Cumulative explained variance plot."""
    if "pca" not in adata.uns or "variance_ratio" not in adata.uns["pca"]:
        raise ValueError("Run PCA first.")
    cum_var = np.cumsum(adata.uns["pca"]["variance_ratio"])
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(range(1, len(cum_var) + 1), cum_var, "o-")
    if threshold is not None:
        ax.axhline(threshold, color="red", linestyle="--", label=f"Threshold={threshold}")
        ax.legend()
    ax.set_xlabel("Number of PCs")
    ax.set_ylabel("Cumulative Variance Explained")
    ax.set_title("Cumulative Variance Explained by PCA")
    plt.tight_layout()
    return fig
