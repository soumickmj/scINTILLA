"""Dunn's post-hoc test after Kruskal-Wallis."""

from __future__ import annotations

from typing import Union

import anndata as ad
import numpy as np
import pandas as pd

from scintilla.io.loaders import ensure_anndata


def dunn_posthoc(
    data: Union[pd.DataFrame, ad.AnnData],
    gene: str,
    group_col: str,
) -> pd.DataFrame:
    """Pairwise Dunn post-hoc test for a single gene.

    Parameters
    ----------
    data:
        Input data.
    gene:
        Gene to test.
    group_col:
        Column in obs with group labels.

    Returns
    -------
    pd.DataFrame with pairwise comparison p-values.
    """
    try:
        import scikit_posthocs as sp  # noqa: PLC0415
    except ImportError as exc:
        raise ImportError(
            "scikit-posthocs is required for Dunn's test. "
            "Install with: pip install scikit-posthocs"
        ) from exc

    adata = ensure_anndata(data)
    if group_col not in adata.obs.columns:
        raise KeyError(f"Column '{group_col}' not found in obs.")
    if gene not in adata.var_names:
        raise KeyError(f"Gene '{gene}' not found in var_names.")

    X = adata.X if not hasattr(adata.X, "toarray") else adata.X.toarray()
    X = X.astype(np.float64)
    gene_idx = list(adata.var_names).index(gene)
    groups = adata.obs[group_col].values

    series = pd.Series(X[:, gene_idx], name=gene)
    group_series = pd.Categorical(groups)
    df = pd.DataFrame({"value": series, "group": group_series})

    result = sp.posthoc_dunn(df, val_col="value", group_col="group", p_adjust="holm")
    return result
