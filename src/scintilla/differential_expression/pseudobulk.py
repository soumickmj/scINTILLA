"""Pseudobulk differential expression."""

from __future__ import annotations

import warnings
from typing import Dict, Optional, Union

import anndata as ad
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests

from scintilla._compat import get_matrix
from scintilla.io.loaders import ensure_anndata

_DEFAULT_PSEUDOCOUNT = 1e-2


def _pseudobulk_table(adata_sub, condition_col, sample_col, pseudocount, layer):
    """Pseudobulk Wilcoxon test of one AnnData (a single cell type or the whole dataset)."""
    X = get_matrix(adata_sub, layer, dense=False, dtype=np.float64)
    conditions = adata_sub.obs[condition_col].values
    samples = adata_sub.obs[sample_col].values

    unique_conds = np.unique(conditions)
    if len(unique_conds) != 2:
        raise ValueError(f"condition_col must have exactly 2 unique values, got {unique_conds}")

    cond1, cond2 = unique_conds[0], unique_conds[1]

    # Aggregate counts per biological sample (the sum of its cells, sparse-safe)
    def _agg(cond):
        mask = conditions == cond
        agg_list = []
        for s in np.unique(samples[mask]):
            smask = mask & (samples == s)
            agg_list.append(np.asarray(X[smask].sum(axis=0)).ravel())
        return np.vstack(agg_list) if agg_list else np.zeros((0, X.shape[1]))

    bulk1 = _agg(cond1)
    bulk2 = _agg(cond2)

    if bulk1.shape[0] < 2 or bulk2.shape[0] < 2:
        raise ValueError(
            "pseudobulk requires at least 2 biological samples per "
            f"condition; got {cond1}={bulk1.shape[0]}, "
            f"{cond2}={bulk2.shape[0]}"
        )

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


def pseudobulk_de(
    adata: Union[ad.AnnData, pd.DataFrame],
    condition_col: str,
    sample_col: str,
    cell_type_col: Optional[str] = None,
    alpha: float = 0.05,
    pseudocount: float = _DEFAULT_PSEUDOCOUNT,
    *,
    layer: Optional[str] = None,
) -> Union[pd.DataFrame, Dict[str, pd.DataFrame]]:
    """Pseudobulk differential expression analysis.

    Sums the counts of each biological sample and runs a Wilcoxon rank-sum test between
    the two conditions.  At least two samples per condition are required: cells are never
    substituted for missing replicates.

    Parameters
    ----------
    adata
        Annotated data matrix with raw counts.
    condition_col
        Column in ``adata.obs`` with condition labels (exactly two unique values).
    sample_col
        Column in ``adata.obs`` with sample identifiers.
    cell_type_col
        Deprecated.  Use :func:`pseudobulk_de_by_celltype`, which always returns a dict;
        passing this argument still returns a dict but emits a ``DeprecationWarning``.
    alpha
        Significance threshold (kept for backwards compatibility; unused).
    pseudocount
        Added to group means before computing log2FC to avoid division by zero
        and inflated fold changes.  Default 1e-2.
    layer
        Layer to aggregate; ``None`` uses ``adata.X``.

    Returns
    -------
    pandas.DataFrame
        One row per gene: ``gene``, ``statistic``, ``p_value``, ``p_adjusted``, ``log2fc``.
    """
    if cell_type_col is not None:
        warnings.warn(
            "pseudobulk_de(cell_type_col=...) returns a dict and is deprecated; "
            "use pseudobulk_de_by_celltype().",
            DeprecationWarning,
            stacklevel=2,
        )
        return pseudobulk_de_by_celltype(
            adata, condition_col, sample_col, cell_type_col, alpha=alpha, pseudocount=pseudocount, layer=layer,
        )
    adata = ensure_anndata(adata)
    return _pseudobulk_table(adata, condition_col, sample_col, pseudocount, layer)


def pseudobulk_de_by_celltype(
    adata: Union[ad.AnnData, pd.DataFrame],
    condition_col: str,
    sample_col: str,
    cell_type_col: str,
    alpha: float = 0.05,
    pseudocount: float = _DEFAULT_PSEUDOCOUNT,
    *,
    layer: Optional[str] = None,
) -> Dict[str, pd.DataFrame]:
    """Run :func:`pseudobulk_de` separately within every cell type.

    A cell type that cannot be tested (for example, fewer than two samples per condition)
    gets an empty table whose ``attrs`` carry ``status="failed"`` and the
    ``failure_reason``, and a warning is emitted; it never disappears from the result.

    Parameters
    ----------
    adata, condition_col, sample_col, alpha, pseudocount, layer
        See :func:`pseudobulk_de`.
    cell_type_col
        Column in ``adata.obs`` with the cell-type labels.

    Returns
    -------
    dict
        ``{cell_type: DataFrame}``.
    """
    adata = ensure_anndata(adata)
    if cell_type_col not in adata.obs.columns:
        raise KeyError(f"Column '{cell_type_col}' not found in obs.")
    results = {}
    for ct in adata.obs[cell_type_col].unique():
        sub = adata[adata.obs[cell_type_col] == ct]
        try:
            results[ct] = _pseudobulk_table(sub, condition_col, sample_col, pseudocount, layer)
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
