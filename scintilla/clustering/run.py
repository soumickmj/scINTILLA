"""Dispatcher for unsupervised analysis pipeline."""

from __future__ import annotations

from typing import Optional, Union

import anndata as ad
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

from scintilla.io.loaders import ensure_anndata
from scintilla.preprocessing.pca import run_pca
from scintilla.clustering.benchmark import benchmark_clustering_methods


def unsupervised_analysis(
    data: Union[pd.DataFrame, ad.AnnData],
    cell_type_col: str,
    use_rep: Optional[str] = None,
    n_clusters: Optional[int] = None,
    run_pca_first: bool = True,
    n_pca_comps: int = 30,
    store_labels: bool = True,
    n_jobs: int = 1,
    verbose: bool = True,
    config=None,
) -> dict:
    """Run a full unsupervised (clustering) analysis pipeline.

    Parameters
    ----------
    data:
        Input data.
    cell_type_col:
        Column in obs with ground-truth cell-type labels.
    use_rep:
        Representation key for neighbour graph. If None and run_pca_first,
        uses 'X_pca'.
    n_clusters:
        Number of clusters (defaults to number of unique cell types).
    run_pca_first:
        Whether to run PCA before clustering.
    n_pca_comps:
        Number of PCA components.
    store_labels:
        If True, store the best method's cluster labels in
        ``adata.obs['scintilla_cluster']``.
    config:
        Optional :class:`~scintilla.analysis_config.AnalysisConfig` to
        control which clustering methods are executed.

    Returns
    -------
    dict with keys: adata, results_df, labels_dict, fig, best_method.
    ``labels_dict`` maps every *method_params* key to its label array.
    When *store_labels* is True the best labels are also in
    ``adata.obs['scintilla_cluster']``.
    """
    adata = ensure_anndata(data)

    if run_pca_first:
        adata = run_pca(adata, n_comps=n_pca_comps)
        rep = "X_pca"
    else:
        rep = use_rep or "X_pca"

    _auto_eps = False
    _adaptive_resolution = False
    if config is not None:
        _auto_eps = getattr(config, "auto_eps", False)
        _adaptive_resolution = getattr(config, "adaptive_resolution", False)

    results_df, labels_dict, fig = benchmark_clustering_methods(
        adata,
        cell_type_col=cell_type_col,
        use_rep=rep,
        n_clusters=n_clusters,
        adaptive_resolution=_adaptive_resolution,
        auto_eps=_auto_eps,
        n_jobs=n_jobs,
        verbose=verbose,
        config=config,
    )

    valid = results_df.dropna(subset=["ari"])
    best_method = valid.sort_values("ari", ascending=False).iloc[0]["method"] if not valid.empty else None

    # Store labels in adata.obs for easy downstream use
    if store_labels and not valid.empty:
        true_labels = adata.obs[cell_type_col].values

        # Build kNN index for confusion scoring
        if rep in adata.obsm:
            X_rep = adata.obsm[rep]
        else:
            X_rep = adata.X if not hasattr(adata.X, "toarray") else adata.X.toarray()
        nn = NearestNeighbors(n_neighbors=50)
        nn.fit(X_rep)
        nn_indices = nn.kneighbors(return_distance=False)

        top = valid.sort_values("ari", ascending=False).head(6)
        for rank, (_, row) in enumerate(top.iterrows(), start=1):
            full_key = f"{row['method']}_{row['params']}"
            if full_key not in labels_dict:
                continue
            method_name = f"{row['method']}|{row['params']}"
            # Store cluster labels with method name in column
            col = f"scintilla_top{rank}_{method_name}"
            adata.obs[col] = pd.Categorical(labels_dict[full_key].astype(str))

            # Compute per-cell confusion score vs cell type labels
            clusters = labels_dict[full_key].astype(str)
            celltypes = true_labels.astype(str)
            k = nn_indices.shape[1]
            scores = np.empty(len(clusters), dtype=np.float64)
            for i, neighbors in enumerate(nn_indices):
                same_cluster = clusters[neighbors] == clusters[i]
                same_type = celltypes[neighbors] == celltypes[i]
                scores[i] = np.sum(same_cluster & ~same_type) / k
            adata.obs[f"scintilla_top{rank}_confusion"] = scores

        # Also keep the overall best as the default column
        best_row = top.iloc[0]
        best_full_key = f"{best_row['method']}_{best_row['params']}"
        if best_full_key in labels_dict:
            adata.obs["scintilla_cluster"] = pd.Categorical(labels_dict[best_full_key].astype(str))

    return {
        "adata": adata,
        "results_df": results_df,
        "labels_dict": labels_dict,
        "fig": fig,
        "best_method": best_method,
    }
