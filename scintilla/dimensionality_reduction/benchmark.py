"""Dimensionality reduction benchmark."""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd


def benchmark_embeddings(
    adata,
    use_rep: str = "X_pca",
    methods: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Benchmark dimensionality reduction methods.

    Parameters
    ----------
    adata:
        AnnData.
    use_rep:
        High-dimensional representation key in obsm.
    methods:
        Methods to try: 'umap', 'tsne', 'diffmap'. Defaults to ['umap', 'tsne'].

    Returns
    -------
    pd.DataFrame  columns=[method, trustworthiness, status]
    """
    if methods is None:
        methods = ["umap", "tsne"]

    from scintilla.evaluation.embedding_metrics import trustworthiness  # noqa: PLC0415

    records = []
    if use_rep in adata.obsm:
        X_high = adata.obsm[use_rep].astype(np.float64)
    else:
        X_high = adata.X if not hasattr(adata.X, "toarray") else adata.X.toarray()
        X_high = X_high.astype(np.float64)

    for method in methods:
        try:
            if method == "umap":
                from scintilla.dimensionality_reduction.umap import run_umap  # noqa: PLC0415
                adata_emb = run_umap(adata, use_rep=use_rep)
                X_low = adata_emb.obsm["X_umap"].astype(np.float64)
            elif method == "tsne":
                from scintilla.dimensionality_reduction.tsne import run_tsne  # noqa: PLC0415
                adata_emb = run_tsne(adata, use_rep=use_rep)
                X_low = adata_emb.obsm.get("X_tsne", adata_emb.obsm.get("tsne")).astype(np.float64)
            elif method == "diffmap":
                from scintilla.dimensionality_reduction.diffusion_map import run_diffusion_map  # noqa: PLC0415
                adata_emb = run_diffusion_map(adata, use_rep=use_rep)
                X_low = adata_emb.obsm["X_diffmap"].astype(np.float64)
            else:
                records.append({"method": method, "trustworthiness": float("nan"), "status": "unknown"})
                continue

            tw = trustworthiness(X_high, X_low[:, :2] if X_low.ndim > 1 else X_low)
            records.append({"method": method, "trustworthiness": tw, "status": "ok"})
        except Exception as e:
            records.append({"method": method, "trustworthiness": float("nan"), "status": str(e)})

    return pd.DataFrame(records)
