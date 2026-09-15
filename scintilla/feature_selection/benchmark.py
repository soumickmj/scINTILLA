"""Feature selection benchmark."""

from __future__ import annotations

from typing import List, Optional, Union

import anndata as ad
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split

from scintilla.config import RANDOM_SEED, DEFAULT_TEST_SIZE
from scintilla.io.loaders import ensure_anndata


def benchmark_feature_selection(
    adata: Union[pd.DataFrame, ad.AnnData],
    target_col: str,
    methods: Optional[List[str]] = None,
    n_features: Optional[int] = None,
    config=None,
    test_size: Optional[float] = None,
    random_state: Optional[int] = None,
) -> pd.DataFrame:
    """Benchmark feature selection methods by downstream classification accuracy.

    Parameters
    ----------
    adata:
        Input data.
    target_col:
        Column in obs with target labels.
    methods:
        List of methods to benchmark.  Defaults to fast methods
        (``pca_loadings``, ``mutual_information``).  Boruta and MRMR
        are available via ``methods=[..., "boruta", "mrmr"]`` but are
        slow on large datasets.
    n_features:
        Number of features to select per method.
    config:
        Optional :class:`~scintilla.analysis_config.AnalysisConfig`.
        If provided, ``config.feature_selection_methods`` and
        ``config.n_features`` are used as defaults.

    Returns
    -------
    pd.DataFrame  columns=[method, n_features, accuracy, f1]
    """
    if methods is None:
        if config is not None and hasattr(config, "feature_selection_methods") and config.feature_selection_methods:
            methods = config.feature_selection_methods
        else:
            methods = ["pca_loadings", "mutual_information"]

    if n_features is None:
        n_features = getattr(config, "n_features", 50) if config is not None else 50
    if test_size is None:
        test_size = getattr(config, "test_size", DEFAULT_TEST_SIZE) if config is not None else DEFAULT_TEST_SIZE
    if random_state is None:
        random_state = getattr(config, "random_seed", RANDOM_SEED) if config is not None else RANDOM_SEED

    adata = ensure_anndata(adata, target_col=target_col)
    X = adata.X if not hasattr(adata.X, "toarray") else adata.X.toarray()
    X = X.astype(np.float64)
    y = adata.obs[target_col].values

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    records = []

    def _eval(X_tr_sel, X_te_sel, method_name):
        try:
            clf = LogisticRegression(max_iter=500, random_state=random_state, solver="lbfgs")
            clf.fit(X_tr_sel, y_tr)
            y_pred = clf.predict(X_te_sel)
            acc = float(accuracy_score(y_te, y_pred))
            f1 = float(f1_score(y_te, y_pred, average="macro", zero_division=0))
            records.append({
                "method": method_name,
                "n_features": X_tr_sel.shape[1],
                "accuracy": acc,
                "f1": f1,
            })
        except Exception as e:
            records.append({"method": method_name, "n_features": n_features, "accuracy": np.nan, "f1": np.nan})

    for method in methods:
        try:
            if method == "pca_loadings":
                from scintilla.feature_selection.pca_loadings import extract_top_genes_per_pc  # noqa: PLC0415
                from sklearn.decomposition import PCA  # noqa: PLC0415
                n_c = min(10, X_tr.shape[0] - 1, X_tr.shape[1] - 1)
                pca = PCA(n_components=n_c, random_state=random_state)
                pca.fit(X_tr)
                # Use top n_features by loading magnitude
                loadings = np.abs(pca.components_).sum(axis=0)
                idx = np.argsort(loadings)[::-1][:n_features]
                _eval(X_tr[:, idx], X_te[:, idx], method)

            elif method == "mutual_information":
                from scintilla.feature_selection.mutual_information import mi_feature_selection  # noqa: PLC0415
                idx, _ = mi_feature_selection(
                    X_tr, y_tr, n_features=n_features, random_state=random_state,
                )
                _eval(X_tr[:, idx], X_te[:, idx], method)

            elif method == "boruta":
                from scintilla.feature_selection.boruta import boruta_selection  # noqa: PLC0415
                mask, _ = boruta_selection(
                    X_tr, y_tr, n_estimators=50, random_state=random_state,
                )
                idx = np.where(mask)[0]
                if len(idx) == 0:
                    idx = np.arange(min(n_features, X_tr.shape[1]))
                _eval(X_tr[:, idx], X_te[:, idx], method)

            elif method == "mrmr":
                from scintilla.feature_selection.mrmr import mrmr_selection  # noqa: PLC0415
                idx, _ = mrmr_selection(
                    X_tr, y_tr, n_features=n_features, random_state=random_state,
                )
                _eval(X_tr[:, idx], X_te[:, idx], method)

            else:
                records.append({"method": method, "n_features": np.nan, "accuracy": np.nan, "f1": np.nan})
        except Exception as e:
            records.append({"method": method, "n_features": n_features, "accuracy": np.nan, "f1": np.nan})

    return pd.DataFrame(records)


# ------------------------------------------------------------------
# HVG sensitivity analysis
# ------------------------------------------------------------------

def hvg_sensitivity_analysis(
    adata: Union[pd.DataFrame, ad.AnnData],
    target_col: str,
    n_top_genes_values: Optional[List[int]] = None,
    clustering_method: str = "leiden",
    resolution: float = 1.0,
    random_state: int = RANDOM_SEED,
) -> pd.DataFrame:
    """Evaluate clustering robustness across different HVG counts.

    For each ``n_top_genes`` value, selects highly variable genes,
    runs PCA → neighbours → clustering → ARI, and checks whether the
    top-ranked HVG count changes the outcome materially.

    Parameters
    ----------
    adata:
        Input AnnData (raw counts or log-normalised).
    target_col:
        Column in obs with ground-truth labels for ARI computation.
    n_top_genes_values:
        List of HVG counts to sweep.  Defaults to
        ``[1000, 2000, 3000, 5000]``.
    clustering_method:
        ``"leiden"`` (default) or ``"louvain"``.
    resolution:
        Community-detection resolution parameter.

    Returns
    -------
    DataFrame with ``n_top_genes``, ``ari``, ``best`` (bool), and
    ``is_stable`` (True when the best ARI is within 5 % of the
    median ARI across all runs).
    """
    import scanpy as sc  # noqa: PLC0415
    from sklearn.metrics import adjusted_rand_score  # noqa: PLC0415

    if n_top_genes_values is None:
        n_top_genes_values = [1000, 2000, 3000, 5000]

    adata = ensure_anndata(adata, target_col=target_col)
    y_true = adata.obs[target_col].values

    records = []
    for n_hvg in n_top_genes_values:
        ad_tmp = adata.copy()
        try:
            sc.pp.highly_variable_genes(ad_tmp, n_top_genes=min(n_hvg, ad_tmp.n_vars), flavor="seurat_v3")
            ad_tmp = ad_tmp[:, ad_tmp.var["highly_variable"]].copy()
            sc.pp.pca(
                ad_tmp,
                n_comps=min(30, ad_tmp.n_vars - 1, ad_tmp.n_obs - 1),
                random_state=random_state,
            )
            sc.pp.neighbors(ad_tmp, use_rep="X_pca", random_state=random_state)
            if clustering_method == "leiden":
                sc.tl.leiden(
                    ad_tmp, resolution=resolution, key_added="cluster",
                    random_state=random_state,
                )
            else:
                sc.tl.louvain(
                    ad_tmp, resolution=resolution, key_added="cluster",
                    random_state=random_state,
                )
            ari = float(adjusted_rand_score(y_true, ad_tmp.obs["cluster"].values))
        except Exception:
            ari = np.nan
        records.append({"n_top_genes": n_hvg, "ari": ari})

    df = pd.DataFrame(records)
    if df["ari"].notna().any():
        best_ari = df["ari"].max()
        df["best"] = df["ari"] == best_ari
        median_ari = df["ari"].median()
        df["is_stable"] = (best_ari - median_ari) / (abs(median_ari) + 1e-10) < 0.05
    else:
        df["best"] = False
        df["is_stable"] = False
    return df
