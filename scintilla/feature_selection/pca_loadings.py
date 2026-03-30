"""Gene selection via PCA loadings."""

from __future__ import annotations

import copy
from typing import Dict, List, Optional, Union

import anndata as ad
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split

from scintilla.config import RANDOM_SEED, DEFAULT_TEST_SIZE
from scintilla.io.loaders import ensure_anndata


def extract_top_genes_per_pc(
    data: Union[pd.DataFrame, ad.AnnData],
    n_per_pc: int = 9,
    n_pcs: Optional[int] = None,
) -> List[str]:
    """Extract top genes (by absolute loading) for each PC and deduplicate.

    Parameters
    ----------
    data:
        AnnData (must have 'X_pca' and PCA varm stored, or gene-space X).
    n_per_pc:
        Number of top genes per PC.
    n_pcs:
        Number of PCs to consider (default: all available).

    Returns
    -------
    Deduplicated list of gene names.
    """
    adata = ensure_anndata(data)

    # Check if PCA loadings are in varm
    if "PCs" in adata.varm:
        loadings = adata.varm["PCs"]  # (n_genes, n_pcs)
    else:
        # Compute PCA
        X = adata.X if not hasattr(adata.X, "toarray") else adata.X.toarray()
        X = X.astype(np.float64)
        n_c = min(50, X.shape[0] - 1, X.shape[1] - 1)
        pca = PCA(n_components=n_c, random_state=RANDOM_SEED)
        pca.fit(X)
        loadings = pca.components_.T  # (n_genes, n_pcs)

    if n_pcs is not None:
        loadings = loadings[:, :n_pcs]

    gene_names = list(adata.var_names)
    selected = []
    for pc in range(loadings.shape[1]):
        abs_load = np.abs(loadings[:, pc])
        top_idx = np.argsort(abs_load)[-n_per_pc:]
        for idx in top_idx:
            g = gene_names[idx]
            if g not in selected:
                selected.append(g)

    return selected


def build_reduced_dataset(
    data: Union[pd.DataFrame, ad.AnnData],
    gene_list: List[str],
) -> ad.AnnData:
    """Subset AnnData to specified genes."""
    adata = ensure_anndata(data)
    valid = [g for g in gene_list if g in adata.var_names]
    if not valid:
        raise ValueError("None of the specified genes found in adata.var_names.")
    return adata[:, valid].copy()


def validate_reduced_set(
    data: Union[pd.DataFrame, ad.AnnData],
    gene_list: List[str],
    target_col: str,
    classifier: str = "qda",
    test_size: float = DEFAULT_TEST_SIZE,
) -> Dict:
    """Compare classification accuracy on full vs reduced gene set.

    Returns
    -------
    dict: accuracy_full, accuracy_reduced, cm_full, cm_reduced
    """
    from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis, LinearDiscriminantAnalysis  # noqa
    from sklearn.ensemble import RandomForestClassifier  # noqa
    from sklearn.metrics import accuracy_score, confusion_matrix  # noqa

    adata = ensure_anndata(data, target_col=target_col)
    if target_col not in adata.obs.columns:
        raise KeyError(f"Column '{target_col}' not found in obs.")

    y = adata.obs[target_col].values

    clf_map = {
        "qda": QuadraticDiscriminantAnalysis(reg_param=0.01),
        "lda": LinearDiscriminantAnalysis(),
        "rf": RandomForestClassifier(n_estimators=100, random_state=RANDOM_SEED),
    }
    clf_cls = clf_map.get(classifier, QuadraticDiscriminantAnalysis(reg_param=0.01))

    def _eval(X):
        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y, test_size=test_size, random_state=RANDOM_SEED, stratify=y
        )
        m = copy.deepcopy(clf_cls)
        m.fit(X_tr, y_tr)
        y_pred = m.predict(X_te)
        return accuracy_score(y_te, y_pred), confusion_matrix(y_te, y_pred)

    X_full = adata.X if not hasattr(adata.X, "toarray") else adata.X.toarray()
    acc_full, cm_full = _eval(X_full.astype(np.float64))

    adata_red = build_reduced_dataset(adata, gene_list)
    X_red = adata_red.X if not hasattr(adata_red.X, "toarray") else adata_red.X.toarray()
    acc_red, cm_red = _eval(X_red.astype(np.float64))

    return {
        "accuracy_full": acc_full,
        "accuracy_reduced": acc_red,
        "cm_full": cm_full,
        "cm_reduced": cm_red,
    }
