"""Batch correction comparison plots."""

from __future__ import annotations

from typing import Optional

import matplotlib.pyplot as plt
import numpy as np

from scintilla.config import RANDOM_SEED


def plot_batch_correction_comparison(
    adata_before,
    adata_after,
    batch_key: str,
    cell_type_key: Optional[str] = None,
    random_state: int = RANDOM_SEED,
) -> plt.Figure:
    """Show data distribution before and after batch correction.

    Uses PCA if UMAP is not available.

    Parameters
    ----------
    adata_before:
        AnnData before correction.
    adata_after:
        AnnData after correction.
    batch_key:
        Column in obs with batch labels.
    cell_type_key:
        Column in obs with cell type labels (optional).

    Returns
    -------
    matplotlib Figure.
    """
    from sklearn.decomposition import PCA  # noqa: PLC0415

    def _get_coords(adata, key):
        if "X_umap" in adata.obsm:
            return adata.obsm["X_umap"][:, :2]
        elif "X_pca" in adata.obsm:
            return adata.obsm["X_pca"][:, :2]
        else:
            X = adata.X if not hasattr(adata.X, "toarray") else adata.X.toarray()
            pca = PCA(n_components=2, random_state=random_state)
            return pca.fit_transform(X.astype(np.float64))

    n_cols = 4 if cell_type_key else 2
    fig, axes = plt.subplots(1, n_cols, figsize=(5 * n_cols, 5))

    datasets = [("Before", adata_before), ("After", adata_after)]
    colour_keys = [batch_key]
    if cell_type_key:
        colour_keys.append(cell_type_key)

    ax_idx = 0
    for label, adata in datasets:
        coords = _get_coords(adata, batch_key)
        for col_key in colour_keys:
            if ax_idx >= len(axes):
                break
            ax = axes[ax_idx]
            ax_idx += 1
            vals = adata.obs[col_key].values
            unique = np.unique(vals)
            cmap = plt.cm.get_cmap("tab20", len(unique))
            for i, v in enumerate(unique):
                mask = vals == v
                ax.scatter(coords[mask, 0], coords[mask, 1], c=[cmap(i)], s=5, alpha=0.5, label=str(v))
            ax.set_title(f"{label} - {col_key}")
            ax.set_xlabel("PC1/UMAP1")
            ax.set_ylabel("PC2/UMAP2")
            if len(unique) <= 10:
                ax.legend(markerscale=3, fontsize=6)

    plt.tight_layout()
    return fig
