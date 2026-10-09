"""Highly variable gene selection."""

from __future__ import annotations

from typing import Optional

import anndata as ad
import pandas as pd

from scintilla._compat import get_matrix, prepare, record_params
from scintilla._logging import logger


def select_hvg(
    adata: ad.AnnData,
    method: str = "seurat_v3",
    n_top_genes: int = 2000,
    span: float = 0.3,
    *,
    layer: Optional[str] = None,
    key_added: str = "highly_variable",
    subset: bool = False,
    copy: bool = False,
) -> Optional[ad.AnnData]:
    """Flag highly variable genes in ``adata.var[key_added]``.

    scanpy runs on a stand-in object, so only ``var[key_added]`` (a boolean) and, for
    the ``seurat_v3`` flavour, ``var[key_added + "_rank"]`` are written to ``adata``.

    Parameters
    ----------
    adata
        Annotated data matrix with raw counts (``seurat_v3`` and ``pearson_residuals``
        expect counts).
    method
        ``"seurat_v3"`` (default), ``"cell_ranger"`` or ``"pearson_residuals"``.  The
        last falls back to ``seurat_v3`` (with a warning) if scanpy's experimental
        implementation is unavailable.
    n_top_genes
        Number of top genes to select (capped at ``n_vars``).
    span
        Loess span for ``seurat_v3``.
    layer
        Layer to use; ``None`` uses ``adata.X``.
    key_added
        Name of the boolean ``var`` column.
    subset
        If ``True`` return a new AnnData restricted to the selected genes (an AnnData
        cannot be resized in place, so ``adata`` itself is left with only the flag).
    copy
        Return a modified copy instead of modifying ``adata`` in place.  Ignored when
        ``subset=True``, which always returns a new object.

    Returns
    -------
    anndata.AnnData or None
        ``None`` when flagging in place; an AnnData when ``copy=True`` or ``subset=True``.
    """
    try:
        import scanpy as sc
    except ImportError as exc:
        raise ImportError("scanpy is required. Install with: pip install scanpy") from exc

    adata, give_back = prepare(adata, copy=copy)
    n_top = min(n_top_genes, adata.n_vars)

    stand_in = ad.AnnData(
        X=get_matrix(adata, layer, dense=False),
        obs=pd.DataFrame(index=adata.obs_names),
        var=pd.DataFrame(index=adata.var_names),
    )
    if method == "pearson_residuals":
        try:
            sc.experimental.pp.highly_variable_genes(stand_in, n_top_genes=n_top, flavor="pearson_residuals")
        except (AttributeError, ImportError, ValueError) as exc:
            logger.warning("pearson_residuals HVG selection unavailable (%s); using seurat_v3", exc)
            sc.pp.highly_variable_genes(stand_in, n_top_genes=n_top, flavor="seurat_v3", span=span)
    elif method == "cell_ranger":
        sc.pp.highly_variable_genes(stand_in, n_top_genes=n_top, flavor="cell_ranger")
    elif method == "seurat_v3":
        sc.pp.highly_variable_genes(stand_in, n_top_genes=n_top, flavor="seurat_v3", span=span)
    else:
        raise ValueError(f"Unknown HVG method {method!r}; expected 'seurat_v3', 'cell_ranger' or 'pearson_residuals'.")

    adata.var[key_added] = stand_in.var["highly_variable"].to_numpy()
    if "highly_variable_rank" in stand_in.var:
        adata.var[key_added + "_rank"] = stand_in.var["highly_variable_rank"].to_numpy()
    record_params(adata, key_added, method=method, n_top_genes=n_top, span=span, layer=layer, key_added=key_added)

    if subset:
        return adata[:, adata.var[key_added].to_numpy()].copy()
    return adata if give_back else None
