"""Per-gene ANOVA / Kruskal-Wallis tests with multiple testing correction."""

from __future__ import annotations

from typing import List, Optional, Union

import anndata as ad
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests

from scintilla.io.loaders import ensure_anndata


def anova_per_gene(
    data: Union[pd.DataFrame, ad.AnnData],
    gene_list: List[str],
    group_col: str,
    method: str = "f_oneway",
    correction: str = "bonferroni",
) -> pd.DataFrame:
    """Run one-way ANOVA (or Kruskal-Wallis) per gene with correction.

    Parameters
    ----------
    data:
        Input data.
    gene_list:
        Genes to test.
    group_col:
        Column in obs with group labels.
    method:
        'f_oneway' (parametric) or 'kruskal' (non-parametric).
    correction:
        Multiple testing correction method (e.g. 'bonferroni', 'fdr_bh').

    Returns
    -------
    pd.DataFrame with columns: Gene, F_statistic, p_value, p_adjusted, significant.
    """
    adata = ensure_anndata(data)
    if group_col not in adata.obs.columns:
        raise KeyError(f"Column '{group_col}' not found in obs.")

    X = adata.X if not hasattr(adata.X, "toarray") else adata.X.toarray()
    X = X.astype(np.float64)
    groups = adata.obs[group_col].values
    unique_groups = np.unique(groups)

    valid_genes = [g for g in gene_list if g in adata.var_names]
    if not valid_genes:
        return pd.DataFrame(columns=["Gene", "F_statistic", "p_value", "p_adjusted", "significant"])

    gene_idx = {g: i for i, g in enumerate(adata.var_names)}
    test_fn = stats.f_oneway if method == "f_oneway" else stats.kruskal

    f_stats, p_vals = [], []
    for gene in valid_genes:
        j = gene_idx[gene]
        group_data = [X[groups == grp, j] for grp in unique_groups if (groups == grp).sum() > 0]
        try:
            f, p = test_fn(*group_data)
        except Exception:
            f, p = np.nan, np.nan
        f_stats.append(float(f) if not np.isnan(f) else np.nan)
        p_vals.append(float(p) if not np.isnan(p) else 1.0)

    # Multiple testing correction
    valid_mask = ~np.isnan(p_vals)
    p_adj = np.full(len(p_vals), np.nan)
    if valid_mask.any():
        p_vals_arr = np.array(p_vals)
        _, p_adj_valid, _, _ = multipletests(p_vals_arr[valid_mask], method=correction)
        p_adj[valid_mask] = p_adj_valid

    return pd.DataFrame({
        "Gene": valid_genes,
        "F_statistic": f_stats,
        "p_value": p_vals,
        "p_adjusted": p_adj.tolist(),
        "significant": (np.array(p_adj) < 0.05).tolist(),
    })
