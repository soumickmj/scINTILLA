"""Gene expression visualisation utilities."""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import anndata as ad

from scintilla.io.loaders import ensure_anndata


def boxplot_genes_by_group(
    data,
    gene_list,
    group_col: str,
    title: str = "Gene Expression",
) -> plt.Figure:
    """Box plots of selected genes split by group."""
    adata = ensure_anndata(data)
    X = adata.X if not hasattr(adata.X, "toarray") else adata.X.toarray()
    df = pd.DataFrame(X, columns=adata.var_names)
    df[group_col] = adata.obs[group_col].values

    valid = [g for g in gene_list if g in df.columns]
    n = len(valid)
    if n == 0:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "No valid genes", ha="center")
        return fig

    cols = min(4, n)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 3))
    axes = np.array(axes).flatten() if n > 1 else [axes]

    for i, gene in enumerate(valid):
        data_long = df[[gene, group_col]].rename(columns={gene: "expression"})
        sns.boxplot(data=data_long, x=group_col, y="expression", ax=axes[i])
        axes[i].set_title(gene)
        axes[i].set_xlabel("")
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle(title)
    plt.tight_layout()
    return fig


def expression_heatmap(
    data,
    gene_list,
    group_col: str,
    title: str = "Expression Heatmap",
) -> plt.Figure:
    """Heatmap of mean expression per group for selected genes."""
    adata = ensure_anndata(data)
    X = adata.X if not hasattr(adata.X, "toarray") else adata.X.toarray()
    df = pd.DataFrame(X, columns=adata.var_names)
    df[group_col] = adata.obs[group_col].values

    valid = [g for g in gene_list if g in df.columns]
    if not valid:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "No valid genes", ha="center")
        return fig

    mean_expr = df.groupby(group_col)[valid].mean()
    fig, ax = plt.subplots(figsize=(max(6, len(valid) * 0.8), max(4, len(mean_expr) * 0.6)))
    sns.heatmap(mean_expr, ax=ax, cmap="viridis", annot=False)
    ax.set_title(title)
    plt.tight_layout()
    return fig
