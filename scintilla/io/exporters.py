"""Exporters for analysis results."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Union

import anndata as ad
import pandas as pd


def save_results_csv(results: pd.DataFrame, path: Union[str, Path], **kwargs) -> None:
    """Save a DataFrame to CSV."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(path, **kwargs)


def save_results_json(results: Dict[str, Any], path: Union[str, Path], indent: int = 2) -> None:
    """Save a dictionary to JSON, converting non-serialisable types."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    def _convert(obj: Any) -> Any:
        import numpy as np  # noqa: PLC0415

        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, pd.DataFrame):
            return obj.to_dict(orient="list")
        return str(obj)

    with open(path, "w") as fh:
        json.dump(results, fh, indent=indent, default=_convert)


def save_anndata(adata: ad.AnnData, path: Union[str, Path]) -> None:
    """Save an AnnData object to .h5ad format."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    adata.write_h5ad(path)
