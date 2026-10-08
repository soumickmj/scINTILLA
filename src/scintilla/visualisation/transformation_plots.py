"""Visualisation of data distributions before and after transformation."""

import matplotlib.pyplot as plt
import numpy as np

from scintilla._compat import get_matrix
from scintilla.io.loaders import ensure_anndata


def before_after_distribution_plot(
    original_data,
    transformed_data,
    transform_name: str = "",
    n_genes: int = 5,
) -> plt.Figure:
    """Side-by-side histograms of n_genes before and after transformation."""
    adata_orig = ensure_anndata(original_data)
    adata_trans = ensure_anndata(transformed_data)

    X_orig = get_matrix(adata_orig, reason="transformation_plots needs a dense matrix")
    X_trans = get_matrix(adata_trans, reason="transformation_plots needs a dense matrix")
    n_genes = min(n_genes, X_orig.shape[1], X_trans.shape[1])
    fig, axes = plt.subplots(n_genes, 2, figsize=(10, n_genes * 2.5))
    if n_genes == 1:
        axes = axes[np.newaxis, :]

    for i in range(n_genes):
        axes[i, 0].hist(X_orig[:, i], bins=30, color="steelblue", alpha=0.7)
        axes[i, 0].set_title(f"Gene {i} – Original")
        axes[i, 1].hist(X_trans[:, i], bins=30, color="tomato", alpha=0.7)
        axes[i, 1].set_title(f"Gene {i} – {transform_name or 'Transformed'}")

    plt.suptitle(f"Distribution Before/After: {transform_name}", fontsize=12)
    plt.tight_layout()
    return fig
