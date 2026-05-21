"""T-test differential expression."""

from __future__ import annotations

from typing import Union

import anndata as ad
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests

from scintilla.io.loaders import ensure_anndata


def ttest_de(
    adata: Union[pd.DataFrame, ad.AnnData],
    group_col: str,
    group1: str,
    group2: str,
    correction: str = "fdr_bh",
    pseudocount: float = 1e-2,
) -> pd.DataFrame:
    """Welch's t-test for differential expression.

    Parameters
    ----------
    adata:
        Input data.
    group_col:
        Column in obs with group labels.
    group1:
        First group.
    group2:
        Second group.
    correction:
        Multiple testing correction method.
    pseudocount:
        Added to group means before computing log2FC to avoid division by zero
        and inflated fold changes.  Default 1e-2.  A small value is used
        instead of the conventional 1.0 (as in DESeq2) because single-cell
        expression means after normalisation are typically in the 0–5 range;
        adding 1.0 would compress true fold-change differences for lowly-
        expressed genes.

    Returns
    -------
    pd.DataFrame  columns=[gene, statistic, p_value, p_adjusted, log2fc]
    """
    adata = ensure_anndata(adata)
    X = adata.X if not hasattr(adata.X, "toarray") else adata.X.toarray()
    X = X.astype(np.float64)
    groups = adata.obs[group_col].values

    mask1 = groups == group1
    mask2 = groups == group2
    if not mask1.any() or not mask2.any():
        raise ValueError(f"Groups '{group1}' or '{group2}' not found.")

    X1 = X[mask1]
    X2 = X[mask2]

    # Vectorised Welch's t-test across all genes at once
    t_stats, p_vals = stats.ttest_ind(X1, X2, axis=0, equal_var=False)
    t_stats = np.where(np.isnan(t_stats), 0.0, t_stats)
    p_vals = np.where(np.isnan(p_vals), 1.0, p_vals)

    # Vectorised log2 fold change
    log2fc = np.log2((X1.mean(axis=0) + pseudocount) / (X2.mean(axis=0) + pseudocount))

    _, p_adj, _, _ = multipletests(p_vals, method=correction)

    # Effect sizes (always included — cheap and critical for interpretation)
    from scintilla.statistical_tests.effect_sizes import cohens_d, hedges_g  # noqa: PLC0415

    d_vals = cohens_d(X1, X2)
    g_vals = hedges_g(X1, X2)

    return pd.DataFrame({
        "gene": list(adata.var_names),
        "statistic": t_stats,
        "p_value": p_vals,
        "p_adjusted": p_adj,
        "log2fc": log2fc,
        "cohens_d": d_vals,
        "hedges_g": g_vals,
    })
