"""Rank genes groups wrapper."""

from __future__ import annotations

from typing import Union

import anndata as ad
import numpy as np
import pandas as pd

from scintilla.io.loaders import ensure_anndata
from scintilla.config import RANDOM_SEED


def rank_genes_groups(
    adata: Union[pd.DataFrame, ad.AnnData],
    groupby: str,
    method: str = "wilcoxon",
    n_genes: int = 50,
    random_state: int = RANDOM_SEED,
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
    random_state:
        Random seed used when ``method="logreg"``.

    Returns
    -------
    pd.DataFrame with columns [group, gene, score, pval, pval_adj, logfoldchange].
    Scanpy's logistic-regression method ranks genes by model coefficient and
    does not calculate p-values or log-fold changes; those columns are ``NaN``
    when ``method="logreg"``.
    """
    try:
        import scanpy as sc  # noqa: PLC0415
    except ImportError as exc:
        raise ImportError("scanpy is required. Install with: pip install scanpy") from exc

    adata = ensure_anndata(adata)
    kwargs = {"random_state": random_state} if method == "logreg" else {}
    sc.tl.rank_genes_groups(
        adata,
        groupby=groupby,
        method=method,
        n_genes=n_genes,
        **kwargs,
    )

    records = []
    result = adata.uns["rank_genes_groups"]
    groups = result["names"].dtype.names
    for group in groups:
        n_available = min(n_genes, len(result["names"][group]))
        for i in range(n_available):
            records.append({
                "group": group,
                "gene": result["names"][group][i],
                "score": result["scores"][group][i],
                "pval": np.nan if method == "logreg" else result["pvals"][group][i],
                "pval_adj": np.nan if method == "logreg" else result["pvals_adj"][group][i],
                "logfoldchange": (
                    np.nan
                    if method == "logreg"
                    else result["logfoldchanges"][group][i]
                ),
            })

    return pd.DataFrame(records, columns=[
        "group", "gene", "score", "pval", "pval_adj", "logfoldchange",
    ])
