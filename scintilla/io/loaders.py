"""Data loaders for various single-cell file formats."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional, Union

import anndata as ad
import numpy as np
import pandas as pd


def load_h5ad(path: Union[str, Path]) -> ad.AnnData:
    """Load an AnnData object from an .h5ad file."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    return ad.read_h5ad(path)


def load_h5mu(path: Union[str, Path]) -> "mudata.MuData":
    """Load a MuData object from an .h5mu file."""
    try:
        import mudata as md
    except ImportError as exc:
        raise ImportError(
            "mudata is required to load .h5mu files. Install with: pip install mudata"
        ) from exc
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    return md.read_h5mu(path)


def load_csv(
    path: Union[str, Path],
    index_col: Optional[int] = 0,
    transpose: bool = False,
    **kwargs,
) -> ad.AnnData:
    """Load a CSV file and return an AnnData object.

    Parameters
    ----------
    path:
        Path to the CSV file.
    index_col:
        Column to use as index. Default 0.
    transpose:
        If True, transpose the DataFrame (genes as rows -> cells as rows).
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    df = pd.read_csv(path, index_col=index_col, **kwargs)
    if transpose:
        df = df.T
    return ensure_anndata(df)


def auto_detect_format(path: Union[str, Path]) -> ad.AnnData:
    """Auto-detect file format and load accordingly."""
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".h5ad":
        return load_h5ad(path)
    elif suffix == ".h5mu":
        return load_h5mu(path)
    elif suffix in (".csv", ".tsv", ".txt"):
        sep = "\t" if suffix in (".tsv", ".txt") else ","
        return load_csv(path, sep=sep)
    else:
        raise ValueError(f"Unsupported file format: {suffix}")


def ensure_anndata(
    data: Union[pd.DataFrame, ad.AnnData, np.ndarray],
    target_col: Optional[str] = None,
) -> ad.AnnData:
    """Convert input data to AnnData if it is not already.

    Numeric columns become the X matrix; non-numeric columns go to obs.
    If *target_col* is provided it is preserved in obs even if numeric.
    """
    if isinstance(data, ad.AnnData):
        return data

    if isinstance(data, np.ndarray):
        return ad.AnnData(X=data.astype(np.float32))

    if isinstance(data, pd.DataFrame):
        # Separate metadata columns from feature columns
        meta_cols = list(data.select_dtypes(exclude=[np.number]).columns)
        if target_col is not None and target_col in data.columns and target_col not in meta_cols:
            meta_cols.append(target_col)

        feature_cols = [c for c in data.columns if c not in meta_cols]
        X = data[feature_cols].values.astype(np.float32)
        obs = data[meta_cols].copy() if meta_cols else pd.DataFrame(index=data.index)
        adata = ad.AnnData(
            X=X,
            obs=obs,
            var=pd.DataFrame(index=feature_cols),
        )
        adata.obs.index = data.index.astype(str)
        return adata

    raise TypeError(f"Cannot convert {type(data)} to AnnData.")
