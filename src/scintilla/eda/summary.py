"""Exploratory data analysis summary utilities.

All functions accept an :class:`~anndata.AnnData` (a DataFrame, array or MuData
is converted by :func:`scintilla.io.ensure_anndata`) and keep sparse matrices
sparse: nothing here allocates a dense copy of ``X``.
"""

from __future__ import annotations

from typing import Optional, Union

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse

from scintilla._compat import get_matrix, sparsity
from scintilla.io.loaders import ensure_anndata


def dataset_summary(
    adata: Union[ad.AnnData, pd.DataFrame],
    *,
    layer: Optional[str] = None,
    modality: Optional[str] = None,
) -> dict:
    """Return a high-level summary of the dataset.

    Parameters
    ----------
    adata
        Annotated data matrix.
    layer
        Layer to summarise; ``None`` uses ``.X``.
    modality
        Modality to use when a MuData is given.

    Returns
    -------
    dict
        ``n_cells``, ``n_genes``, ``total_counts``, ``mean_counts_per_cell``,
        ``sparsity`` (fraction of zero entries, computed from the number of stored
        non-zeros, so a CSR matrix is never densified), ``obs_columns`` and
        ``var_columns``.
    """
    adata = ensure_anndata(adata, modality=modality)
    X = get_matrix(adata, layer, dense=False)
    total_counts = float(X.sum())
    return {
        "n_cells": adata.n_obs,
        "n_genes": adata.n_vars,
        "total_counts": total_counts,
        "mean_counts_per_cell": total_counts / max(adata.n_obs, 1),
        "sparsity": sparsity(X),
        "obs_columns": list(adata.obs.columns),
        "var_columns": list(adata.var.columns),
    }


def expressed_genes(
    adata: Union[ad.AnnData, pd.DataFrame],
    min_cells: int = 1,
    *,
    layer: Optional[str] = None,
    modality: Optional[str] = None,
) -> list[str]:
    """Return genes expressed (value > 0) in at least ``min_cells`` cells.

    Parameters
    ----------
    adata
        Annotated data matrix.
    min_cells
        Minimum number of cells in which a gene must be expressed.
    layer
        Layer to use; ``None`` uses ``.X``.
    modality
        Modality to use when a MuData is given.
    """
    adata = ensure_anndata(adata, modality=modality)
    X = get_matrix(adata, layer, dense=False)
    n_cells = np.asarray((X > 0).sum(axis=0)).ravel()
    return list(adata.var_names[n_cells >= min_cells])


def sample_counts(
    adata: Union[ad.AnnData, pd.DataFrame],
    group_col: str,
    *,
    modality: Optional[str] = None,
) -> pd.Series:
    """Return cell counts per group.

    Parameters
    ----------
    adata
        Annotated data matrix.
    group_col
        Column of ``adata.obs`` defining the groups.
    modality
        Modality to use when a MuData is given.
    """
    adata = ensure_anndata(adata, modality=modality)
    if group_col not in adata.obs.columns:
        raise KeyError(f"Column '{group_col}' not found in obs.")
    return adata.obs[group_col].value_counts()


def mean_expression_by_group(
    adata: Union[ad.AnnData, pd.DataFrame],
    group_col: str,
    gene_list: Optional[list[str]] = None,
    *,
    layer: Optional[str] = None,
    modality: Optional[str] = None,
) -> pd.DataFrame:
    """Compute mean expression per group for the requested genes.

    Only the requested genes are materialised, and the group means of a sparse
    matrix are obtained with a sparse indicator product, so memory scales with the
    number of selected genes, not with ``n_obs x n_vars``.

    Parameters
    ----------
    adata
        Annotated data matrix.
    group_col
        Column of ``adata.obs`` defining the groups.
    gene_list
        Genes to keep; unknown names are ignored.  ``None`` keeps all genes.
    layer
        Layer to use; ``None`` uses ``.X``.
    modality
        Modality to use when a MuData is given.

    Returns
    -------
    pandas.DataFrame
        Groups (sorted) by genes.
    """
    adata = ensure_anndata(adata, modality=modality)
    if group_col not in adata.obs.columns:
        raise KeyError(f"Column '{group_col}' not found in obs.")
    if gene_list:
        keep = [g for g in gene_list if g in adata.var_names]
        view = adata[:, keep]
    else:
        view = adata
    X = get_matrix(view, layer, dense=False)
    codes, groups = pd.factorize(adata.obs[group_col].to_numpy(), sort=True)
    valid = codes >= 0
    indicator = sparse.csr_matrix(
        (np.ones(valid.sum()), (codes[valid], np.flatnonzero(valid))),
        shape=(len(groups), adata.n_obs),
    )
    sums = indicator @ X
    sums = sums.toarray() if sparse.issparse(sums) else np.asarray(sums)
    counts = np.asarray(indicator.sum(axis=1)).ravel()
    means = sums / counts[:, None]
    return pd.DataFrame(means, index=pd.Index(groups, name=group_col), columns=view.var_names)
