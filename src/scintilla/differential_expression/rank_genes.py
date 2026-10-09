"""Rank genes groups wrapper."""

from __future__ import annotations

from typing import Optional, Union

import anndata as ad
import numpy as np
import pandas as pd

from scintilla._compat import get_matrix
from scintilla.config import RANDOM_SEED
from scintilla.io.loaders import ensure_anndata


def rank_genes_groups(
    adata: Union[ad.AnnData, pd.DataFrame],
    groupby: str,
    method: str = "wilcoxon",
    n_genes: int = 50,
    random_state: int = RANDOM_SEED,
    *,
    layer: Optional[str] = None,
    key_added: Optional[str] = None,
) -> pd.DataFrame:
    """Rank genes per group with scanpy and return a tidy table.

    scanpy runs on a stand-in object, so ``adata`` gains no ``uns["rank_genes_groups"]``
    entry unless ``key_added`` is given.

    Parameters
    ----------
    adata
        Annotated data matrix (normalised, log-transformed expression expected).
    groupby
        Column in ``adata.obs`` with group labels.
    method
        DE method for scanpy (``"wilcoxon"``, ``"t-test"``, ``"logreg"``, ...).
    n_genes
        Number of top genes to return per group.
    random_state
        Random seed used when ``method="logreg"``.
    layer
        Layer to use; ``None`` uses ``adata.X``.
    key_added
        If given, also store the table in ``adata.uns[key_added]``.

    Returns
    -------
    pandas.DataFrame
        Columns ``group``, ``gene``, ``score``, ``pval``, ``pval_adj`` and
        ``logfoldchange``.  Scanpy's logistic-regression method ranks genes by model
        coefficient and does not calculate p-values or log-fold changes; those columns
        are ``NaN`` when ``method="logreg"``.
    """
    try:
        import scanpy as sc
    except ImportError as exc:
        raise ImportError("scanpy is required. Install with: pip install scanpy") from exc

    source = ensure_anndata(adata)
    if groupby not in source.obs.columns:
        raise KeyError(f"Column '{groupby}' not found in obs.")
    adata = ad.AnnData(
        X=get_matrix(source, layer, dense=False),
        obs=source.obs[[groupby]].copy(),
        var=pd.DataFrame(index=source.var_names),
    )
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

    table = pd.DataFrame(records, columns=[
        "group", "gene", "score", "pval", "pval_adj", "logfoldchange",
    ])
    if key_added is not None:
        source.uns[key_added] = table
    return table
