"""Rank genes groups wrapper."""

from __future__ import annotations

from typing import Union

import anndata as ad
import numpy as np
import pandas as pd

from scintilla.io.loaders import ensure_anndata


def rank_genes_groups(
    adata: Union[pd.DataFrame, ad.AnnData],
    groupby: str,
    method: str = "wilcoxon",
    n_genes: int = 50,
) -> pd.DataFrame:
    """Wrapper around scanpy.tl.rank_genes_groups.

    Parameters
    ----------
    adata:
        Input data.
    groupby:
        Column in obs with group labels.
    method:
        DE method for scanpy ('wilcoxon', 't-test', 'logreg', etc.).
    n_genes:
        Number of top genes to return per group.

    Returns
    -------
    pd.DataFrame with columns [group, gene, score, pval, pval_adj, logfoldchange]
    """
    try:
        import scanpy as sc  # noqa: PLC0415
    except ImportError as exc:
        raise ImportError("scanpy is required. Install with: pip install scanpy") from exc

    adata = ensure_anndata(adata)
    sc.tl.rank_genes_groups(adata, groupby=groupby, method=method, n_genes=n_genes)

    records = []
    groups = adata.uns["rank_genes_groups"]["names"].dtype.names
    for group in groups:
        for i in range(n_genes):
            try:
                records.append({
                    "group": group,
                    "gene": adata.uns["rank_genes_groups"]["names"][group][i],
                    "score": adata.uns["rank_genes_groups"]["scores"][group][i],
                    "pval": adata.uns["rank_genes_groups"]["pvals"][group][i],
                    "pval_adj": adata.uns["rank_genes_groups"]["pvals_adj"][group][i],
                    "logfoldchange": adata.uns["rank_genes_groups"]["logfoldchanges"][group][i],
                })
            except (IndexError, KeyError):
                break

    return pd.DataFrame(records)
