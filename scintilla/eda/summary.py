"""Exploratory data analysis summary utilities."""

from __future__ import annotations

from typing import List, Optional, Union

import anndata as ad
import numpy as np
import pandas as pd

from scintilla.io.loaders import ensure_anndata


def dataset_summary(data: Union[pd.DataFrame, ad.AnnData]) -> dict:
    """Return a high-level summary of the dataset."""
    adata = ensure_anndata(data)
    X = adata.X if not hasattr(adata.X, "toarray") else adata.X.toarray()
    total_counts = float(np.sum(X))
    sparsity = float(np.mean(X == 0))
    return {
        "n_cells": adata.n_obs,
        "n_genes": adata.n_vars,
        "total_counts": total_counts,
        "mean_counts_per_cell": total_counts / max(adata.n_obs, 1),
        "sparsity": sparsity,
        "obs_columns": list(adata.obs.columns),
        "var_columns": list(adata.var.columns),
    }


def expressed_genes(
    data: Union[pd.DataFrame, ad.AnnData],
    min_cells: int = 1,
) -> List[str]:
    """Return genes expressed in at least *min_cells* cells."""
    adata = ensure_anndata(data)
    X = adata.X if not hasattr(adata.X, "toarray") else adata.X.toarray()
    mask = np.sum(X > 0, axis=0) >= min_cells
    return list(adata.var_names[mask])


def sample_counts(
    data: Union[pd.DataFrame, ad.AnnData],
    group_col: str,
) -> pd.Series:
    """Return cell counts per group."""
    adata = ensure_anndata(data)
    if group_col not in adata.obs.columns:
        raise KeyError(f"Column '{group_col}' not found in obs.")
    return adata.obs[group_col].value_counts()


def mean_expression_by_group(
    data: Union[pd.DataFrame, ad.AnnData],
    group_col: str,
    gene_list: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Compute mean expression per group for specified genes."""
    adata = ensure_anndata(data)
    if group_col not in adata.obs.columns:
        raise KeyError(f"Column '{group_col}' not found in obs.")
    X = adata.X if not hasattr(adata.X, "toarray") else adata.X.toarray()
    df = pd.DataFrame(X, index=adata.obs_names, columns=adata.var_names)
    df[group_col] = adata.obs[group_col].values
    if gene_list:
        cols = [g for g in gene_list if g in df.columns] + [group_col]
        df = df[cols]
    return df.groupby(group_col).mean()
