"""Patient-level pseudo-bulk aggregation."""

from __future__ import annotations

from typing import Union

import anndata as ad
import numpy as np
import pandas as pd

from scintilla.io.loaders import ensure_anndata


def aggregate_by_patient_celltype(
    data: Union[pd.DataFrame, ad.AnnData],
    patient_col: str,
    celltype_col: str,
    method: str = "median",
) -> pd.DataFrame:
    """Aggregate single-cell data to patient x cell-type pseudo-bulk profiles.

    Parameters
    ----------
    data:
        Input data.
    patient_col:
        Column in obs containing patient identifiers.
    celltype_col:
        Column in obs containing cell-type labels.
    method:
        Aggregation method: 'median' or 'mean'.

    Returns
    -------
    pd.DataFrame with MultiIndex (patient, cell_type) and gene columns.
    """
    adata = ensure_anndata(data)
    X = adata.X if not hasattr(adata.X, "toarray") else adata.X.toarray()
    df = pd.DataFrame(X, columns=adata.var_names, index=adata.obs_names)
    for col in [patient_col, celltype_col]:
        if col not in adata.obs.columns:
            raise KeyError(f"Column '{col}' not found in obs.")
    df[patient_col] = adata.obs[patient_col].values
    df[celltype_col] = adata.obs[celltype_col].values

    agg_fn = "median" if method == "median" else "mean"
    result = df.groupby([patient_col, celltype_col]).agg(agg_fn)
    return result
