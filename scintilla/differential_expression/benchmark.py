"""DE benchmark: compare multiple DE methods."""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from scintilla.io.loaders import ensure_anndata


def benchmark_de_methods(
    adata,
    condition_col: str,
    group1: str,
    group2: str,
    n_genes: int = 100,
) -> pd.DataFrame:
    """Benchmark DE methods by comparing results.

    Parameters
    ----------
    adata:
        Input data.
    condition_col:
        Column in obs with condition labels.
    group1:
        First group label.
    group2:
        Second group label.
    n_genes:
        Number of top genes to compare.

    Returns
    -------
    pd.DataFrame  columns=[method, n_sig_genes, n_genes_tested, status]
    """
    adata = ensure_anndata(adata)
    records = []

    methods = {
        "wilcoxon": _run_wilcoxon,
        "ttest": _run_ttest,
        "pseudobulk": _run_pseudobulk,
    }

    for name, fn in methods.items():
        try:
            result = fn(adata, condition_col, group1, group2)
            n_sig = int((result["p_adjusted"] < 0.05).sum())
            records.append({
                "method": name,
                "n_sig_genes": n_sig,
                "n_genes_tested": len(result),
                "status": "ok",
            })
        except Exception as e:
            records.append({"method": name, "n_sig_genes": np.nan, "n_genes_tested": np.nan, "status": str(e)})

    return pd.DataFrame(records)


def _run_wilcoxon(adata, condition_col, group1, group2):
    from scintilla.differential_expression.wilcoxon import wilcoxon_de  # noqa: PLC0415
    return wilcoxon_de(adata, condition_col, group1, group2)


def _run_ttest(adata, condition_col, group1, group2):
    from scintilla.differential_expression.ttest import ttest_de  # noqa: PLC0415
    return ttest_de(adata, condition_col, group1, group2)


def _run_pseudobulk(adata, condition_col, group1, group2):
    from scintilla.differential_expression.pseudobulk import pseudobulk_de  # noqa: PLC0415
    # Need a sample_col - create dummy if not available
    if "sample" not in adata.obs.columns:
        import anndata as ad  # noqa: PLC0415
        adata = adata.copy()
        groups = adata.obs[condition_col].values
        adata.obs["sample"] = groups + "_s1"
    return pseudobulk_de(adata, condition_col, "sample")
