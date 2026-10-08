"""Marker-based cell type annotation."""

from __future__ import annotations

from typing import Dict, List, Optional

import anndata as ad
import numpy as np
import pandas as pd

from scintilla._compat import finish, get_matrix, prepare, record_params
from scintilla._logging import logger
from scintilla.config import RANDOM_SEED


def annotate_by_markers(
    adata: ad.AnnData,
    marker_dict: Dict[str, List[str]],
    method: str = "score",
    threshold: float = 0.1,
    random_state: int = RANDOM_SEED,
    *,
    layer: Optional[str] = None,
    key_added: str = "predicted_cell_type",
    copy: bool = False,
) -> Optional[ad.AnnData]:
    """Annotate cells by marker gene sets.

    The winning cell type per cell is written to ``adata.obs[key_added]`` and the
    per-type scores to ``adata.obsm[key_added + "_scores"]`` (a DataFrame whose columns
    are the cell types).  Nothing else is added to ``adata``.

    Parameters
    ----------
    adata
        Annotated data matrix (normalised, log-transformed expression for ``"score"``).
    marker_dict
        Dictionary ``{cell_type: [marker_genes]}``.  Genes absent from ``adata`` are
        ignored; a cell type with no usable marker scores 0.
    method
        ``"score"``: scanpy ``score_genes`` per cell type, falling back to the mean
        expression of the markers when the control gene pool is too small.
        ``"threshold"``: fraction of markers expressed above ``threshold``.
    threshold
        Expression threshold for the ``"threshold"`` method.
    random_state
        Random seed used by scanpy marker scoring.
    layer
        Layer to score; ``None`` uses ``adata.X``.
    key_added
        Name of the ``obs`` column that receives the predicted cell type.
    copy
        Return a modified copy instead of modifying ``adata`` in place.

    Returns
    -------
    anndata.AnnData or None
        ``None`` when working in place; the modified copy when ``copy=True``.
    """
    adata, give_back = prepare(adata, copy=copy)
    cell_types = list(marker_dict.keys())
    if method not in {"score", "threshold"}:
        raise ValueError(f"Unknown method {method!r}; expected 'score' or 'threshold'.")

    if method == "score":
        try:
            import scanpy as sc
        except ImportError as exc:
            raise ImportError("scanpy is required. Install with: pip install scanpy") from exc

        X = get_matrix(adata, layer, dense=False)
        # Score on a stand-in so the caller's obs is not littered with score columns.
        work = ad.AnnData(
            X=X, obs=pd.DataFrame(index=adata.obs_names), var=pd.DataFrame(index=adata.var_names)
        )
        scores = np.zeros((adata.n_obs, len(cell_types)))
        for k, (ct, markers) in enumerate(marker_dict.items()):
            valid_markers = [m for m in markers if m in adata.var_names]
            if not valid_markers:
                logger.warning("no markers of %r found in adata.var_names; its score is 0", ct)
                continue
            col = f"score_{k}"
            try:
                sc.tl.score_genes(
                    work,
                    gene_list=valid_markers,
                    score_name=col,
                    ctrl_as_ref=False,
                    random_state=random_state,
                )
                scores[:, k] = work.obs[col].to_numpy()
            except RuntimeError as exc:
                logger.warning("score_genes failed for %r (%s); using mean marker expression", ct, exc)
                idx = [adata.var_names.get_loc(m) for m in valid_markers]
                scores[:, k] = np.asarray(X[:, idx].mean(axis=1)).ravel()
    else:
        X = get_matrix(adata, layer, dense=False)
        gene_idx = {g: i for i, g in enumerate(adata.var_names)}
        scores = np.zeros((adata.n_obs, len(cell_types)))
        for k, markers in enumerate(marker_dict.values()):
            idx = [gene_idx[m] for m in markers if m in gene_idx]
            if idx:
                scores[:, k] = np.asarray((X[:, idx] > threshold).mean(axis=1)).ravel()

    best_idx = scores.argmax(axis=1)
    adata.obs[key_added] = [cell_types[i] for i in best_idx]
    adata.obsm[key_added + "_scores"] = pd.DataFrame(scores, index=adata.obs_names, columns=cell_types)
    record_params(
        adata, "annotate_by_markers", method=method, threshold=threshold, layer=layer,
        random_state=random_state, key_added=key_added, cell_types=cell_types,
    )
    return finish(adata, give_back)
