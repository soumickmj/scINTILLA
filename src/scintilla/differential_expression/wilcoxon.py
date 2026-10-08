"""Wilcoxon rank-sum differential expression test."""

from __future__ import annotations

from typing import Optional, Union

import anndata as ad
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests

from scintilla.differential_expression._common import two_group_matrices
from scintilla.io.loaders import ensure_anndata


def wilcoxon_de(
    adata: Union[pd.DataFrame, ad.AnnData],
    group_col: str,
    group1: str,
    group2: str,
    correction: str = "fdr_bh",
    pseudocount: float = 1e-2,
    *,
    layer: Optional[str] = None,
) -> pd.DataFrame:
    """Wilcoxon rank-sum test for differential expression.

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

    layer:
        Layer to test; ``None`` uses ``adata.X``.  Only the cells of the two groups are
        densified.

    Returns
    -------
    pandas.DataFrame
        One row per gene: ``gene``, ``statistic``, ``p_value``, ``p_adjusted``,
        ``log2fc`` and, where defined, the effect sizes of the test.
    """
    adata = ensure_anndata(adata)
    X1, X2 = two_group_matrices(adata, group_col, group1, group2, layer)

    # Vectorised Mann-Whitney U across all genes (scipy >= 1.8)
    stats_arr, p_vals = stats.mannwhitneyu(X1, X2, axis=0, alternative="two-sided")
    stats_arr = np.where(np.isnan(stats_arr), 0.0, stats_arr)
    p_vals = np.where(np.isnan(p_vals), 1.0, p_vals)

    # Vectorised log2 fold change
    log2fc = np.log2((X1.mean(axis=0) + pseudocount) / (X2.mean(axis=0) + pseudocount))

    _, p_adj, _, _ = multipletests(p_vals, method=correction)

    # Effect sizes (always included — cheap and critical for interpretation)
    from scintilla.statistical_tests.effect_sizes import rank_biserial

    n1, n2 = X1.shape[0], X2.shape[0]
    # Negate rank_biserial so that all directional measures (log2fc,
    # rank_biserial, cliffs_delta) are positive when group1 > group2.
    # The raw formula r = 1 - 2U/(n1*n2) is negative when group1
    # dominates because scipy returns U for group1 which is large in
    # that case.  Cliff's delta = 2U/(n1*n2) - 1 is already positive
    # when group1 dominates, so we negate rank_biserial to align.
    r_rb = -rank_biserial(stats_arr, n1, n2)
    # Cliff's delta derived from U: delta = 2*U/(n1*n2) - 1
    cliffs_d = 2.0 * stats_arr / (n1 * n2) - 1.0

    return pd.DataFrame({
        "gene": list(adata.var_names),
        "statistic": stats_arr,
        "p_value": p_vals,
        "p_adjusted": p_adj,
        "log2fc": log2fc,
        "rank_biserial": r_rb,
        "cliffs_delta": cliffs_d,
    })
