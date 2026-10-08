"""AnnData-level feature selection: flag genes in ``adata.var``."""

from __future__ import annotations

from typing import Optional

import anndata as ad
import numpy as np

from scintilla._compat import finish, get_matrix, prepare, record_params
from scintilla.config import RANDOM_SEED

METHODS = ("pca_loadings", "mutual_information", "boruta", "mrmr")


def select_features(
    adata: ad.AnnData,
    target_col: str,
    method: str = "mutual_information",
    n_features: int = 50,
    random_state: int = RANDOM_SEED,
    *,
    layer: Optional[str] = None,
    key_added: str = "selected",
    copy: bool = False,
) -> Optional[ad.AnnData]:
    """Flag the genes that best separate the labels in ``adata.var[key_added]``.

    Parameters
    ----------
    adata
        Annotated data matrix (normalised expression).
    target_col
        Column in ``adata.obs`` with the labels (not used by ``"pca_loadings"``).
    method
        ``"mutual_information"`` (default), ``"pca_loadings"``, ``"boruta"`` or ``"mrmr"``;
        the last two need the optional ``Boruta`` and ``mrmr-selection`` packages.
    n_features
        Number of genes to select (for ``"pca_loadings"``, genes per PC are taken until
        this many unique genes are found).
    random_state
        Random seed.
    layer
        Layer to use; ``None`` uses ``adata.X``.
    key_added
        Name of the boolean ``var`` column.  Where the method gives a relevance score it
        is stored in ``var[key_added + "_score"]``.
    copy
        Return a modified copy instead of modifying ``adata`` in place.

    Returns
    -------
    anndata.AnnData or None
        ``None`` when working in place; the modified copy when ``copy=True``.
    """
    if method not in METHODS:
        raise ValueError(f"Unknown method {method!r}; expected one of {METHODS}.")
    adata, give_back = prepare(adata, copy=copy)
    if method != "pca_loadings" and target_col not in adata.obs.columns:
        raise KeyError(f"Column '{target_col}' not found in obs.")

    n_vars = adata.n_vars
    mask = np.zeros(n_vars, dtype=bool)
    score = np.full(n_vars, np.nan)

    if method == "pca_loadings":
        from scintilla.feature_selection.pca_loadings import extract_top_genes_per_pc

        genes = extract_top_genes_per_pc(adata, n_per_pc=max(1, n_features), random_state=random_state)
        chosen = adata.var_names.get_indexer(genes[:n_features])
        mask[chosen[chosen >= 0]] = True
    else:
        X = get_matrix(adata, layer, reason=f"{method} feature selection needs a dense matrix", dtype=np.float64)
        y = adata.obs[target_col].to_numpy()
        if method == "mutual_information":
            from scintilla.feature_selection.mutual_information import mi_feature_selection

            idx, scores_df = mi_feature_selection(X, y, n_features=n_features, random_state=random_state)
            mask[idx] = True
            score[scores_df["feature_idx"].to_numpy()] = scores_df["mi_score"].to_numpy()
        elif method == "boruta":
            from scintilla.feature_selection.boruta import boruta_selection

            selected, importances = boruta_selection(X, y, random_state=random_state)
            mask[:] = np.asarray(selected, dtype=bool)
            score[:] = np.asarray(importances, dtype=float)
        else:
            from scintilla.feature_selection.mrmr import mrmr_selection

            idx, scores = mrmr_selection(X, y, n_features=n_features, random_state=random_state)
            mask[np.asarray(idx, dtype=int)] = True
            score[np.asarray(idx, dtype=int)] = np.asarray(scores, dtype=float)

    adata.var[key_added] = mask
    adata.var[key_added + "_score"] = score
    record_params(
        adata, "select_features", target_col=target_col, method=method, n_features=n_features,
        layer=layer, random_state=random_state, key_added=key_added,
    )
    return finish(adata, give_back)
