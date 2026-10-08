"""Marker-based cell type annotation."""

from __future__ import annotations

from typing import Dict, List

import anndata as ad
import numpy as np

from scintilla.config import RANDOM_SEED


def annotate_by_markers(
    adata: ad.AnnData,
    marker_dict: Dict[str, List[str]],
    method: str = "score",
    threshold: float = 0.1,
    random_state: int = RANDOM_SEED,
) -> ad.AnnData:
    """Annotate cells by marker gene sets.

    Parameters
    ----------
    adata:
        Input AnnData.
    marker_dict:
        Dictionary {cell_type: [marker_genes]}.
    method:
        'score': scanpy sc.tl.score_genes per cell type.
        'threshold': fraction of markers expressed above threshold.
    threshold:
        Expression threshold for 'threshold' method.
    random_state:
        Random seed used by scanpy marker scoring.

    Returns
    -------
    AnnData with 'predicted_cell_type' in obs.
    """
    try:
        import scanpy as sc  # noqa: PLC0415
    except ImportError as exc:
        raise ImportError("scanpy is required. Install with: pip install scanpy") from exc

    adata = adata.copy()
    cell_types = list(marker_dict.keys())

    if method == "score":
        score_cols = []
        for ct, markers in marker_dict.items():
            valid_markers = [m for m in markers if m in adata.var_names]
            if not valid_markers:
                adata.obs[f"score_{ct}"] = 0.0
                score_cols.append(f"score_{ct}")
                continue
            col = f"score_{ct}"
            try:
                sc.tl.score_genes(
                    adata,
                    gene_list=valid_markers,
                    score_name=col,
                    ctrl_as_ref=False,
                    random_state=random_state,
                )
            except RuntimeError:
                # Fallback to simple mean expression if gene pool too small
                X = adata.X if not hasattr(adata.X, "toarray") else adata.X.toarray()
                gene_idx = {g: i for i, g in enumerate(adata.var_names)}
                idx = [gene_idx[m] for m in valid_markers if m in gene_idx]
                adata.obs[col] = X[:, idx].mean(axis=1) if idx else 0.0
            score_cols.append(col)

        scores = adata.obs[score_cols].values
        best_idx = scores.argmax(axis=1)
        adata.obs["predicted_cell_type"] = [cell_types[i] for i in best_idx]

    elif method == "threshold":
        X = adata.X if not hasattr(adata.X, "toarray") else adata.X.toarray()
        gene_idx = {g: i for i, g in enumerate(adata.var_names)}
        scores = np.zeros((adata.n_obs, len(cell_types)))
        for k, (ct, markers) in enumerate(marker_dict.items()):
            valid = [m for m in markers if m in gene_idx]
            if valid:
                idx = [gene_idx[m] for m in valid]
                expressed = (X[:, idx] > threshold).mean(axis=1)
                scores[:, k] = expressed
        best_idx = scores.argmax(axis=1)
        adata.obs["predicted_cell_type"] = [cell_types[i] for i in best_idx]

    return adata
