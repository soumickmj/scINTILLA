"""Shared helpers behind the AnnData-first API contract.

Every public function follows the same conventions, and these helpers are the
single place they are implemented:

* :func:`resolve_adata` and :func:`finish` implement ``copy``;
* :func:`get_matrix` is the only place where an expression matrix is
  densified, and says so when it does;
* :func:`record_params` writes provenance to ``adata.uns["scintilla"]``.
"""

from __future__ import annotations

import warnings
from typing import Any

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse

from scintilla._logging import logger

UNS_KEY = "scintilla"


def resolve_adata(data: Any, *, copy: bool = False, target_col: str | None = None, modality: str | None = None) -> ad.AnnData:
    """Return the AnnData to work on.

    ``data`` may be an AnnData (the documented contract), or, as a convenience,
    a DataFrame, an array or a MuData (see :func:`scintilla.io.ensure_anndata`).
    With ``copy=True`` a copy is returned so that the caller's object is
    untouched; otherwise an AnnData input is returned as is and modified in place.
    """
    from scintilla.io.loaders import ensure_anndata

    adata = ensure_anndata(data, target_col=target_col, modality=modality)
    return adata.copy() if copy else adata


def prepare(
    data: Any,
    *,
    copy: bool = False,
    target_col: str | None = None,
    modality: str | None = None,
) -> tuple[ad.AnnData, bool]:
    """Resolve ``data`` to an AnnData and decide whether the caller gets it back.

    Returns ``(adata, give_back)``.  ``give_back`` is ``True`` when ``copy=True`` or
    when the input was a DataFrame or array (the converted AnnData is the only
    handle the caller has), and ``False`` for the in-place case, where scanpy
    convention is to return ``None``.
    """
    adata = resolve_adata(data, copy=copy, target_col=target_col, modality=modality)
    converted = isinstance(data, (pd.DataFrame, np.ndarray))
    return adata, bool(copy or converted)


def finish(adata: ad.AnnData, give_back: bool) -> ad.AnnData | None:
    """Return ``adata`` when it should be handed back, else ``None`` (scanpy convention)."""
    return adata if give_back else None


def get_matrix(
    adata: ad.AnnData,
    layer: str | None = None,
    *,
    dense: bool = True,
    dtype: Any = None,
    reason: str | None = None,
):
    """Return ``adata.X`` or ``adata.layers[layer]``.

    Parameters
    ----------
    adata
        The AnnData.
    layer
        Layer to read; ``None`` reads ``.X``.
    dense
        If ``False`` a sparse matrix stays sparse.  If ``True`` it is densified,
        and ``reason`` (the algorithm that requires it) is logged at debug level.
    dtype
        Optional dtype to cast to.
    reason
        Why a dense array is needed; only used for the debug message.
    """
    X = adata.X if layer is None else adata.layers[layer]
    if sparse.issparse(X):
        if not dense:
            return X.astype(dtype) if dtype is not None else X
        _warn_if_large(X, dtype, reason)
        if reason:
            logger.debug("densifying %s x %s matrix: %s", X.shape[0], X.shape[1], reason)
        # Cast while still sparse, so only one dense copy is ever allocated.
        return (X.astype(dtype) if dtype is not None else X).toarray()
    elif hasattr(X, "toarray"):  # other sparse-like backends
        X = X.toarray()
    X = np.asarray(X)
    return X.astype(dtype) if dtype is not None else X


def _warn_if_large(X: Any, dtype: Any, reason: str | None) -> None:
    """Warn before allocating a dense copy larger than ``settings.dense_warning_gb``."""
    from scintilla.settings import settings

    itemsize = np.dtype(dtype).itemsize if dtype is not None else X.dtype.itemsize
    size_gb = X.shape[0] * X.shape[1] * itemsize / 1024**3
    if size_gb > settings.dense_warning_gb:
        warnings.warn(
            f"Densifying a {X.shape[0]} x {X.shape[1]} sparse matrix allocates {size_gb:.1f} GB"
            + (f" ({reason})" if reason else "")
            + ". Work on a low-dimensional representation (use_rep='X_pca', or "
            "supervised_use_rep='X_pca' in label_quality) or subsample the cells. "
            "Raise scintilla.settings.dense_warning_gb to silence this warning.",
            UserWarning,
            stacklevel=4,
        )


def as_dense(X: Any, dtype: Any = None) -> np.ndarray:
    """Densify an arbitrary matrix-like and optionally cast it."""
    if sparse.issparse(X) or hasattr(X, "toarray"):
        X = X.toarray()
    X = np.asarray(X)
    return X.astype(dtype) if dtype is not None else X


def sparsity(X: Any) -> float:
    """Fraction of zero entries, computed without densifying a sparse matrix."""
    if sparse.issparse(X):
        total = X.shape[0] * X.shape[1]
        return 1.0 - X.count_nonzero() / total if total else 0.0
    return float(np.mean(np.asarray(X) == 0))


def _storable(value: Any) -> Any:
    """Convert a parameter value into something ``write_h5ad`` can store.

    ``None`` is dropped by the caller (h5ad cannot store it), numpy scalars become
    Python scalars, homogeneous lists stay lists, and anything else becomes its ``repr``.
    """
    if isinstance(value, dict):
        return {str(k): _storable(v) for k, v in value.items() if v is not None}
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (np.ndarray, pd.Series)):
        value = value.tolist()
    if isinstance(value, tuple):
        value = list(value)
    if isinstance(value, list):
        items = [_storable(v) for v in value]
        kinds = {type(v) for v in items}
        if items and len(kinds) == 1 and kinds <= {int, float, str, bool}:
            return items
        return repr(value)
    if isinstance(value, (str, int, float, bool)):
        return value
    return repr(value)


def record_params(adata: ad.AnnData, name: str, **params: Any) -> None:
    """Store the parameters that produced a result in ``adata.uns["scintilla"][name]``.

    Parameters set to ``None`` are omitted (they are the defaults and h5ad cannot
    store ``None``).  The record survives :meth:`~anndata.AnnData.write_h5ad`.
    """
    store = adata.uns.setdefault(UNS_KEY, {})
    store[name] = {"params": _storable({k: v for k, v in params.items() if v is not None})}
