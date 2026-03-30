"""Dispatcher for unsupervised analysis pipeline."""

from __future__ import annotations

from typing import Optional, Union

import anndata as ad
import pandas as pd

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

    results_df, labels_dict, fig = benchmark_clustering_methods(
        adata,
        cell_type_col=cell_type_col,
        use_rep=rep,
        n_clusters=n_clusters,
        verbose=verbose,
        config=config,
    )

    valid = results_df.dropna(subset=["ari"])
    best_method = valid.sort_values("ari", ascending=False).iloc[0]["method"] if not valid.empty else None

    # Store best labels in adata.obs for easy downstream use
    if store_labels and best_method is not None:
        best_key = valid.sort_values("ari", ascending=False).iloc[0]
        full_key = f"{best_key['method']}_{best_key['params']}"
        if full_key in labels_dict:
            adata.obs["scintilla_cluster"] = labels_dict[full_key]
        elif labels_dict:
            # fallback: first matching key
            for k, v in labels_dict.items():
                if k.startswith(best_method):
                    adata.obs["scintilla_cluster"] = v
                    break

    return {
        "adata": adata,
        "results_df": results_df,
        "labels_dict": labels_dict,
        "fig": fig,
        "best_method": best_method,
    }
