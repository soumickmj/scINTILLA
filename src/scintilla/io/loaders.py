"""Data loaders for various single-cell file formats."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Optional, Union

import anndata as ad
import numpy as np
import pandas as pd

if TYPE_CHECKING:
    import mudata


def load_h5ad(path: Union[str, Path]) -> ad.AnnData:
    """Load an AnnData object from an .h5ad file.

    Parameters
    ----------
    path
        Destination path.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    return ad.read_h5ad(path)


def load_h5mu(path: Union[str, Path]) -> mudata.MuData:
    """Load a MuData object from an .h5mu file.

    Parameters
    ----------
    path
        Destination path.
    """
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


def auto_detect_format(path: Union[str, Path]) -> Union[ad.AnnData, mudata.MuData]:
    """Auto-detect the file format from the suffix and load it.

    ``.h5ad`` and CSV/TSV files give an :class:`~anndata.AnnData`; ``.h5mu`` gives a
    :class:`~mudata.MuData`, which every analysis function accepts together with a
    ``modality`` argument (see :func:`ensure_anndata`).

    Parameters
    ----------
    path
        Destination path.
    """
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
    adata: Union[pd.DataFrame, ad.AnnData, np.ndarray, mudata.MuData],
    target_col: Optional[str] = None,
    modality: Optional[str] = None,
) -> ad.AnnData:
    """Convert input data to AnnData if it is not already.

    AnnData is returned unchanged (not copied).  A DataFrame is converted with its
    numeric columns as ``X`` and non-numeric columns as ``obs``; *target_col* is
    kept in ``obs`` even if numeric.  A MuData is reduced to one modality:
    *modality* selects it, and may be omitted only when the MuData has exactly one.

    Parameters
    ----------
    data
        AnnData (the documented contract), DataFrame, 2-D array or MuData.
    target_col
        Column to preserve in ``obs`` when converting a DataFrame.
    modality
        Modality to extract from a MuData, for example ``"rna"``.

    Returns
    -------
    anndata.AnnData
    """
    if isinstance(adata, ad.AnnData):
        return adata

    if type(adata).__module__.split(".")[0] == "mudata":
        modalities = list(adata.mod.keys())
        if modality is None:
            if len(modalities) != 1:
                raise ValueError(
                    f"MuData holds several modalities {modalities}; pass modality=... to choose one."
                )
            modality = modalities[0]
        if modality not in adata.mod:
            raise KeyError(f"Modality {modality!r} not in MuData; available: {modalities}")
        return adata.mod[modality]

    if isinstance(adata, np.ndarray):
        return ad.AnnData(X=adata.astype(np.float32))

    if isinstance(adata, pd.DataFrame):
        # Separate metadata columns from feature columns
        meta_cols = list(adata.select_dtypes(exclude=[np.number]).columns)
        if target_col is not None and target_col in adata.columns and target_col not in meta_cols:
            meta_cols.append(target_col)

        feature_cols = [c for c in adata.columns if c not in meta_cols]
        X = adata[feature_cols].values.astype(np.float32)
        obs = adata[meta_cols].copy() if meta_cols else pd.DataFrame(index=adata.index)
        adata = ad.AnnData(
            X=X,
            obs=obs,
            var=pd.DataFrame(index=feature_cols),
        )
        adata.obs.index = adata.index.astype(str)
        return adata

    raise TypeError(f"Cannot convert {type(adata)} to AnnData.")
