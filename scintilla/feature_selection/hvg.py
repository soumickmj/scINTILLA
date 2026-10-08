"""Highly variable gene selection."""

from __future__ import annotations

from typing import Union

import anndata as ad
import numpy as np

from scintilla.io.loaders import ensure_anndata


def select_hvg(
    adata: ad.AnnData,
    method: str = "seurat_v3",
    n_top_genes: int = 2000,
    span: float = 0.3,
) -> ad.AnnData:
    """Select highly variable genes.

    Parameters
    ----------
    adata:
        AnnData object with raw or normalised counts.
    method:
        HVG selection method: 'seurat_v3', 'cell_ranger', or 'pearson_residuals'.
    n_top_genes:
        Number of top HVGs to select.
    span:
        Loess span for 'seurat_v3'.

    Returns
    -------
    AnnData filtered to HVGs only.
    """
    try:
        import scanpy as sc  # noqa: PLC0415
    except ImportError as exc:
        raise ImportError("scanpy is required. Install with: pip install scanpy") from exc

    adata = ensure_anndata(adata)
    n_top = min(n_top_genes, adata.n_vars)

    if method == "pearson_residuals":
        try:
            sc.experimental.pp.highly_variable_genes(
                adata, n_top_genes=n_top, flavor="pearson_residuals"
            )
        except Exception:
            # Fallback to seurat_v3
            sc.pp.highly_variable_genes(adata, n_top_genes=n_top, flavor="seurat_v3", span=span)
    elif method == "cell_ranger":
        sc.pp.highly_variable_genes(adata, n_top_genes=n_top, flavor="cell_ranger")
    else:  # seurat_v3
        sc.pp.highly_variable_genes(adata, n_top_genes=n_top, flavor="seurat_v3", span=span)

    return adata[:, adata.var["highly_variable"]].copy()
