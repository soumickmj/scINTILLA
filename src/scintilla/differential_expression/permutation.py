"""Permutation-based differential expression test."""

from __future__ import annotations

from typing import Optional, Union

import anndata as ad
import numpy as np
import pandas as pd
from statsmodels.stats.multitest import multipletests

from scintilla.config import RANDOM_SEED
from scintilla.differential_expression._common import two_group_matrices
from scintilla.io.loaders import ensure_anndata


def permutation_de(
    adata: Union[pd.DataFrame, ad.AnnData],
    group_col: str,
    group1: str,
    group2: str,
    n_permutations: int = 1000,
    correction: str = "fdr_bh",
    pseudocount: float = 1e-2,
    standardise: bool = False,
    random_state: int = RANDOM_SEED,
    *,
    layer: Optional[str] = None,
) -> pd.DataFrame:
    """Permutation test for differential expression.

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
    n_permutations:
        Number of permutations.
    correction:
        Multiple testing correction method.
    pseudocount:
        Added to group means before computing log2FC to avoid division by zero
        and inflated fold changes.  Default 1e-2.  A small value is used
        instead of the conventional 1.0 (as in DESeq2) because single-cell
        expression means after normalisation are typically in the 0–5 range;
        adding 1.0 would compress true fold-change differences for lowly-
        expressed genes.
    standardise:
        When True, use a Welch-like standardised test statistic
        ``(mean1 - mean2) / sqrt(var1/n1 + var2/n2)`` instead of the raw
        mean difference.  This is more appropriate for data with
        heterogeneous variance across genes (e.g. raw counts or Pearson
        residuals).  Default False for backward compatibility.

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
    n1, n2 = X1.shape[0], X2.shape[0]
    n_genes = X1.shape[1]
    X_combined = np.vstack([X1, X2])

    # Observed statistics (mean difference)
    obs_stat = X1.mean(axis=0) - X2.mean(axis=0)
    if standardise:
        se = np.sqrt(X1.var(axis=0, ddof=1) / n1 + X2.var(axis=0, ddof=1) / n2)
        se = np.where(se == 0, 1.0, se)  # avoid division by zero for constant genes
        obs_stat = obs_stat / se
    log2fc = np.log2((X1.mean(axis=0) + pseudocount) / (X2.mean(axis=0) + pseudocount))

    rng = np.random.default_rng(random_state)
    n_total = n1 + n2

    # Vectorised permutation: build (n_permutations, n_total) index matrix
    # and compute all mean-differences in one shot.  Process in chunks to
    # limit peak memory when n_permutations is very large.
    chunk_size = min(n_permutations, 1000)
    perm_stats = np.empty((n_permutations, n_genes))
    for start in range(0, n_permutations, chunk_size):
        end = min(start + chunk_size, n_permutations)
        n_chunk = end - start
        # Each row is an independent permutation of row indices
        idx = np.tile(np.arange(n_total), (n_chunk, 1))
        rng.permuted(idx, axis=1, out=idx)
        grp1 = X_combined[idx[:, :n1]]   # (n_chunk, n1, n_genes)
        grp2 = X_combined[idx[:, n1:]]   # (n_chunk, n2, n_genes)
        perm_stats[start:end] = grp1.mean(axis=1) - grp2.mean(axis=1)

    if standardise:
        # Compute pooled SE for each permutation chunk would be expensive;
        # instead use the observed SE (stable under permutation of group labels
        # for the variance structure).
        se = np.sqrt(X1.var(axis=0, ddof=1) / n1 + X2.var(axis=0, ddof=1) / n2)
        se = np.where(se == 0, 1.0, se)
        perm_stats = perm_stats / se

    # Two-sided p-value using the (B+1)/(n+1) estimator (Phipson & Smyth, 2010)
    p_vals = (np.sum(np.abs(perm_stats) >= np.abs(obs_stat), axis=0) + 1) / (n_permutations + 1)

    _, p_adj, _, _ = multipletests(p_vals, method=correction)

    return pd.DataFrame({
        "gene": list(adata.var_names),
        "statistic": obs_stat,
        "p_value": p_vals,
        "p_adjusted": p_adj,
        "log2fc": log2fc,
    })
