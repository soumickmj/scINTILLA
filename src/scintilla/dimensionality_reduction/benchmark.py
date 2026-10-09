"""Dimensionality reduction benchmark."""

from __future__ import annotations

from typing import List, Optional

import numpy as np
import pandas as pd

from scintilla._logging import logger
from scintilla.config import RANDOM_SEED
from scintilla.io.loaders import ensure_anndata


def benchmark_embeddings(
    adata,
    use_rep: str = "X_pca",
    methods: Optional[List[str]] = None,
    random_state: int = RANDOM_SEED,
) -> pd.DataFrame:
    """Benchmark dimensionality reduction methods.

    Parameters
    ----------
    adata
        Annotated data matrix.  It is not modified.
    use_rep
        High-dimensional representation key in obsm (a temporary PCA is used when
        absent).
    methods
        Methods to try: ``"umap"``, ``"tsne"``, ``"diffmap"``.  Defaults to
        ``["umap", "tsne"]``.
    random_state
        Random seed.

    Returns
    -------
    pandas.DataFrame
        Columns ``method``, ``trustworthiness`` and ``status``.  A method that fails
        appears with ``status`` set to the error message, never silently dropped.
    """
    if methods is None:
        methods = ["umap", "tsne"]

    from scintilla.dimensionality_reduction._common import get_representation, stand_in
    from scintilla.dimensionality_reduction.diffusion_map import run_diffusion_map
    from scintilla.dimensionality_reduction.tsne import run_tsne
    from scintilla.dimensionality_reduction.umap import run_umap
    from scintilla.evaluation.embedding_metrics import trustworthiness

    records = []
    adata = ensure_anndata(adata)
    rep = get_representation(adata, use_rep, random_state)
    X_high = rep.astype(np.float64)  # for the trustworthiness score only; embed the representation as stored
    # Embed a stand-in that holds only the representation: the caller's object is untouched.
    work = stand_in(rep, adata.obs_names)

    runners = {
        "umap": (run_umap, "X_umap"),
        "tsne": (run_tsne, "X_tsne"),
        "diffmap": (run_diffusion_map, "X_diffmap"),
    }
    for method in methods:
        if method not in runners:
            records.append({"method": method, "trustworthiness": float("nan"), "status": "unknown"})
            continue
        runner, key = runners[method]
        try:
            runner(work, use_rep="rep", random_state=random_state)
            X_low = work.obsm[key].astype(np.float64)
            tw = trustworthiness(X_high, X_low[:, :2] if X_low.ndim > 1 else X_low)
            records.append({"method": method, "trustworthiness": tw, "status": "ok"})
        except Exception as e:  # reported in the ``status`` column, never dropped
            logger.warning("%s embedding failed: %s", method, e)
            records.append({"method": method, "trustworthiness": float("nan"), "status": str(e)})

    return pd.DataFrame(records)
