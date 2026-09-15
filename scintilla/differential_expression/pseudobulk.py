"""Pseudobulk differential expression."""

from __future__ import annotations

import warnings
from typing import Dict, Optional, Union

import anndata as ad
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests

from scintilla.io.loaders import ensure_anndata

_DEFAULT_PSEUDOCOUNT = 1e-2


def pseudobulk_de(
    adata: Union[pd.DataFrame, ad.AnnData],
    condition_col: str,
    sample_col: str,
    cell_type_col: Optional[str] = None,
    alpha: float = 0.05,
    pseudocount: float = _DEFAULT_PSEUDOCOUNT,
) -> Union[pd.DataFrame, Dict[str, pd.DataFrame]]:
    """Pseudobulk differential expression analysis.

    Aggregates counts per sample then runs Wilcoxon rank-sum test.

    Parameters
    ----------
    adata:
        Input data.
    condition_col:
        Column in obs with condition labels (must have exactly 2 unique values).
    sample_col:
        Column in obs with sample IDs.
    cell_type_col:
        If provided, runs DE per cell type.
    alpha:
        Significance threshold.
    pseudocount:
        Added to group means before computing log2FC to avoid division by zero
        and inflated fold changes.  Default 1e-2.

    Returns
    -------
    pd.DataFrame or dict {cell_type: pd.DataFrame} with DE results.
    """
    adata = ensure_anndata(adata)

    def _run_de(adata_sub):
        X = adata_sub.X if not hasattr(adata_sub.X, "toarray") else adata_sub.X.toarray()
        X = X.astype(np.float64)
        conditions = adata_sub.obs[condition_col].values
        samples = adata_sub.obs[sample_col].values

        unique_conds = np.unique(conditions)
        if len(unique_conds) != 2:
            raise ValueError(f"condition_col must have exactly 2 unique values, got {unique_conds}")

        cond1, cond2 = unique_conds[0], unique_conds[1]

        # Aggregate per sample
        def _agg(cond):
            mask = conditions == cond
            samps = np.unique(samples[mask])
            agg_list = []
            for s in samps:
                smask = mask & (samples == s)
                agg_list.append(X[smask].sum(axis=0))
            return np.vstack(agg_list) if agg_list else np.zeros((0, X.shape[1]))

        bulk1 = _agg(cond1)
        bulk2 = _agg(cond2)

        if bulk1.shape[0] < 2 or bulk2.shape[0] < 2:
            raise ValueError(
                "pseudobulk requires at least 2 biological samples per "
                f"condition; got {cond1}={bulk1.shape[0]}, "
                f"{cond2}={bulk2.shape[0]}"
            )

        n_genes = X.shape[1]

        # Vectorised Mann-Whitney U across all genes (scipy >= 1.8)
        t_stats, p_vals = stats.mannwhitneyu(
            bulk1, bulk2, axis=0, alternative="two-sided",
        )
        t_stats = np.where(np.isnan(t_stats), 0.0, t_stats)
        p_vals = np.where(np.isnan(p_vals), 1.0, p_vals)

        # Vectorised log2 fold change
        log2fc = np.log2(
            (bulk1.mean(axis=0) + pseudocount) / (bulk2.mean(axis=0) + pseudocount)
        )

        _, p_adj, _, _ = multipletests(p_vals, method="fdr_bh")

        return pd.DataFrame({
            "gene": list(adata_sub.var_names),
            "statistic": t_stats,
            "p_value": p_vals,
            "p_adjusted": p_adj,
            "log2fc": log2fc,
        })

    if cell_type_col is None:
        return _run_de(adata)

    results = {}
    for ct in adata.obs[cell_type_col].unique():
        sub = adata[adata.obs[cell_type_col] == ct].copy()
        try:
            results[ct] = _run_de(sub)
        except MemoryError:
            raise
        except Exception as exc:
            warnings.warn(
                f"Pseudobulk DE failed for cell type {ct!r}: {exc}",
                UserWarning,
                stacklevel=2,
            )
            failed = pd.DataFrame(columns=[
                "gene", "statistic", "p_value", "p_adjusted", "log2fc",
            ])
            failed.attrs.update(status="failed", failure_reason=str(exc))
            results[ct] = failed
    return results
