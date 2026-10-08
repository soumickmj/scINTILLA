"""Input and output handling shared by the array-level clustering algorithms."""

from __future__ import annotations

from typing import Any, Optional

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse

from scintilla._compat import record_params
from scintilla.io.loaders import ensure_anndata


def resolve_matrix(data: Any, use_rep: Optional[str] = None, *, dense: bool = False) -> np.ndarray:
    """Return the float64 matrix to cluster.

    ``data`` may be an AnnData (``obsm[use_rep]`` if ``use_rep`` is given, else ``X``), a
    DataFrame, or a NumPy/SciPy matrix.  Sparse input stays sparse unless ``dense=True``.
    """
    if isinstance(data, (pd.DataFrame, ad.AnnData)):
        adata = ensure_anndata(data)
        if use_rep is not None:
            if use_rep not in adata.obsm:
                raise KeyError(f"adata.obsm has no '{use_rep}'")
            X = adata.obsm[use_rep]
        else:
            X = adata.X
    else:
        X = data
    if sparse.issparse(X) or hasattr(X, "toarray"):
        return X.toarray().astype(np.float64) if dense else X.astype(np.float64)
    return np.asarray(X, dtype=np.float64)


def store_labels(data: Any, labels: np.ndarray, key_added: Optional[str], method: str, **params: Any) -> None:
    """Write ``labels`` to ``data.obs[key_added]`` (a categorical) with provenance, if requested."""
    if key_added is None:
        return
    if not isinstance(data, ad.AnnData):
        raise TypeError("key_added requires an AnnData input.")
    data.obs[key_added] = pd.Categorical(np.asarray(labels).astype(str))
    record_params(data, key_added, method=method, **params)
