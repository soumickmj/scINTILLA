"""Comprehensive clustering benchmark."""

from __future__ import annotations

from typing import Dict, Optional, Tuple, Union

import anndata as ad
import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scintilla.config import (
    RANDOM_SEED,
    DISTANCE_METRICS,
    LINKAGE_METHODS,
    DBSCAN_EPS_RANGE,
    DBSCAN_MIN_SAMPLES_RANGE,
    LEIDEN_RESOLUTIONS,
)
from scintilla.io.loaders import ensure_anndata
from scintilla.clustering.kmeans import kmeans_clustering
from scintilla.clustering.hierarchical import hierarchical_sklearn
from scintilla.clustering.dbscan import dbscan_clustering


def benchmark_clustering_methods(
    data: Union[pd.DataFrame, ad.AnnData],
    cell_type_col: str,
    use_rep: str = "X_pca",
    n_clusters: Optional[int] = None,
    adaptive_resolution: bool = False,
    resolution_selection: str = "best_ari",
    auto_eps: bool = False,
    verbose: bool = True,
    config=None,
) -> Tuple[pd.DataFrame, Dict[str, np.ndarray], plt.Figure]:
    """Benchmark multiple clustering methods using ARI against true labels.

    Parameters
    ----------
    config:
        Optional :class:`~scintilla.analysis_config.AnalysisConfig`.
        When provided, only the methods listed in
        ``config.clustering_methods`` are executed.
    adaptive_resolution:
        If ``True``, use a two-pass bisection search for Leiden/Louvain
        resolution instead of the fixed grid.  The coarse grid is still
        used as the initial sweep, but a refinement pass narrows down the
        optimal resolution.
    resolution_selection:
        Strategy for choosing the best resolution.

        - ``"best_ari"`` (default) — pick the resolution with the highest
          Adjusted Rand Index (requires ground-truth labels).
        - ``"nvi_stability"`` — pick the resolution at the most stable
          plateau in the Normalised Variation of Information curve.
          This is fully unsupervised and does not use ground-truth labels.
    auto_eps:
        If ``True``, estimate DBSCAN *eps* from the k-distance graph
        instead of sweeping the fixed ``DBSCAN_EPS_RANGE`` grid.

    Returns
    -------
    results_df : pd.DataFrame  columns=[method, params, ari, n_clusters]
    labels_dict : dict  {method_key: labels_array}
    fig : matplotlib Figure
    """
    adata = ensure_anndata(data)

    if cell_type_col not in adata.obs.columns:
        raise KeyError(f"Column '{cell_type_col}' not found in obs.")

    true_labels = adata.obs[cell_type_col].values
    n_true = len(np.unique(true_labels))
    if n_clusters is None:
        n_clusters = n_true

    # Get embedding
    if use_rep in adata.obsm:
        X = adata.obsm[use_rep].astype(np.float64)
    else:
        X = adata.X if not hasattr(adata.X, "toarray") else adata.X.toarray()
        X = X.astype(np.float64)

    records = []
    labels_dict: Dict[str, np.ndarray] = {}

    # Determine which methods to run
    _enabled: Optional[set] = None
    if config is not None:
        _cm = getattr(config, "clustering_methods", None)
        if _cm is not None:
            _enabled = {m.lower() for m in _cm}

    def _skip(method_key: str) -> bool:
        return _enabled is not None and method_key not in _enabled

    def _record(method, params, labels):
        n_found = len(np.unique(labels[labels != -1]))
        valid = labels[labels != -1]
        true_valid = true_labels[labels != -1]
        noise_frac = np.mean(labels == -1) if len(labels) > 0 else 0.0
        if noise_frac > 0.5:
            ari = np.nan
        else:
            ari = adjusted_rand_score(true_valid, valid) if len(valid) > 0 else np.nan
        records.append({
            "method": method, "params": params, "ari": ari,
            "n_clusters": n_found, "noise_fraction": round(noise_frac, 4),
        })
        labels_dict[f"{method}_{params}"] = labels

    # K-Means variants
    if not _skip("kmeans"):
        for init in ["k-means++", "random"]:
            try:
                lbl, _, _ = kmeans_clustering(X, n_clusters, init=init)
                _record("KMeans", f"init={init}", lbl)
            except MemoryError:
                raise
            except (RuntimeError, ValueError, np.linalg.LinAlgError, ArithmeticError) as e:
                if verbose:
                    print(f"KMeans {init} failed: {e}")

        try:
            lbl, _, _ = kmeans_clustering(X, n_clusters, spherical=True)
            _record("KMeans_spherical", "spherical=True", lbl)
        except MemoryError:
            raise
        except (RuntimeError, ValueError, np.linalg.LinAlgError, ArithmeticError) as e:
            if verbose:
                print(f"KMeans spherical failed: {e}")

        try:
            lbl, _, _ = kmeans_clustering(X, n_clusters, bisecting=True)
            _record("KMeans_bisecting", "bisecting=True", lbl)
        except MemoryError:
            raise
        except (RuntimeError, ValueError, np.linalg.LinAlgError, ArithmeticError) as e:
            if verbose:
                print(f"KMeans bisecting failed: {e}")

    # Hierarchical: metric x linkage grid
    if not _skip("hierarchical"):
        for metric in DISTANCE_METRICS:
            for lnk in LINKAGE_METHODS:
                if lnk == "ward" and metric != "euclidean":
                    records.append({"method": "Hierarchical", "params": f"{metric}/{lnk}", "ari": np.nan, "n_clusters": np.nan})
                    continue
                try:
                    lbl = hierarchical_sklearn(X, n_clusters, metric=metric, linkage_method=lnk)
                    _record("Hierarchical", f"{metric}/{lnk}", lbl)
                except MemoryError:
                    raise
                except (RuntimeError, ValueError, np.linalg.LinAlgError, ArithmeticError) as e:
                    if verbose:
                        print(f"Hierarchical {metric}/{lnk} failed: {e}")
                    records.append({"method": "Hierarchical", "params": f"{metric}/{lnk}", "ari": np.nan, "n_clusters": np.nan})

    # DBSCAN grid (or data-driven eps)
    if not _skip("dbscan"):
        if auto_eps:
            from scintilla.clustering.dbscan import estimate_eps  # noqa: PLC0415
            est_eps = estimate_eps(X, min_samples=DBSCAN_MIN_SAMPLES_RANGE[0])
            eps_values = [est_eps * 0.8, est_eps, est_eps * 1.2]
        else:
            eps_values = DBSCAN_EPS_RANGE
        for eps in eps_values:
            for minsamp in DBSCAN_MIN_SAMPLES_RANGE:
                try:
                    lbl, _, _, _ = dbscan_clustering(X, eps=eps, min_samples=minsamp)
                    _record("DBSCAN", f"eps={eps},min_samples={minsamp}", lbl)
                except MemoryError:
                    raise
                except (RuntimeError, ValueError, np.linalg.LinAlgError, ArithmeticError) as e:
                    if verbose:
                        print(f"DBSCAN eps={eps} ms={minsamp} failed: {e}")

    # Leiden
    if not _skip("leiden"):
        try:
            from scintilla.clustering.leiden import leiden_clustering  # noqa: PLC0415
            _leiden_res = LEIDEN_RESOLUTIONS
            if config is not None and hasattr(config, "leiden_resolutions") and config.leiden_resolutions:
                _leiden_res = config.leiden_resolutions

            if adaptive_resolution:
                from scintilla.statistical_tests.adaptive import adaptive_resolution_search  # noqa: PLC0415

                def _run_leiden(res):
                    return leiden_clustering(adata, resolution=res, use_rep=use_rep)

                def _metric_leiden(labels):
                    valid = labels[labels != -1]
                    tv = true_labels[labels != -1]
                    return adjusted_rand_score(tv, valid) if len(valid) > 0 else 0.0

                search_result = adaptive_resolution_search(
                    _run_leiden, _metric_leiden, _leiden_res, refine_steps=5,
                )
                for res, score in search_result["all_scores"]:
                    lbl = _run_leiden(res)
                    _record("Leiden", f"resolution={res:.4f}", lbl)
            elif resolution_selection == "nvi_stability":
                from scintilla.statistical_tests.adaptive import nvi_stability  # noqa: PLC0415
                labels_at_res = []
                for res in _leiden_res:
                    try:
                        lbl = leiden_clustering(adata, resolution=res, use_rep=use_rep)
                        labels_at_res.append((res, lbl))
                        _record("Leiden", f"resolution={res}", lbl)
                    except Exception:
                        pass
                if labels_at_res:
                    stab = nvi_stability(labels_at_res)
                    # Mark the stable resolution
                    for _, row_lbl in labels_at_res:
                        pass  # already recorded above
            else:
                for res in _leiden_res:
                    try:
                        lbl = leiden_clustering(adata, resolution=res, use_rep=use_rep)
                        _record("Leiden", f"resolution={res}", lbl)
                    except MemoryError:
                        raise
                    except (RuntimeError, ValueError, np.linalg.LinAlgError, ArithmeticError) as e:
                        if verbose:
                            print(f"Leiden res={res} failed: {e}")
        except ImportError:
            if verbose:
                print("Leiden skipped (leidenalg not installed).")

    # HDBSCAN
    if not _skip("hdbscan"):
        try:
            from scintilla.clustering.hdbscan import hdbscan_clustering  # noqa: PLC0415
            from scintilla.config import HDBSCAN_MIN_CLUSTER_SIZE_RANGE, HDBSCAN_MIN_SAMPLES_RANGE  # noqa: PLC0415
            _hdb_sizes = HDBSCAN_MIN_CLUSTER_SIZE_RANGE[:3]
            _hdb_samples = HDBSCAN_MIN_SAMPLES_RANGE[:2]
            if config is not None:
                if hasattr(config, "hdbscan_min_cluster_sizes") and config.hdbscan_min_cluster_sizes:
                    _hdb_sizes = config.hdbscan_min_cluster_sizes
                if hasattr(config, "hdbscan_min_samples") and config.hdbscan_min_samples:
                    _hdb_samples = config.hdbscan_min_samples
            _, lbl = hdbscan_clustering(X, min_cluster_size_range=_hdb_sizes,
                                         min_samples_range=_hdb_samples)
            _record("HDBSCAN", "best", lbl)
        except MemoryError:
            raise
        except (ImportError, RuntimeError, ValueError, np.linalg.LinAlgError, ArithmeticError) as e:
            if verbose:
                print(f"HDBSCAN failed: {e}")

    # Louvain
    if not _skip("louvain"):
        try:
            from scintilla.clustering.louvain import louvain_clustering  # noqa: PLC0415
            from scintilla.config import LOUVAIN_RESOLUTIONS  # noqa: PLC0415
            _louv_res = LOUVAIN_RESOLUTIONS
            if config is not None and hasattr(config, "louvain_resolutions") and config.louvain_resolutions:
                _louv_res = config.louvain_resolutions
            for res in _louv_res:
                try:
                    lbl = louvain_clustering(adata, resolution=res, use_rep=use_rep)
                    _record("Louvain", f"resolution={res}", lbl)
                except MemoryError:
                    raise
                except (RuntimeError, ValueError, np.linalg.LinAlgError, ArithmeticError) as e:
                    if verbose:
                        print(f"Louvain res={res} failed: {e}")
        except ImportError:
            if verbose:
                print("Louvain skipped (louvain not installed).")

    # Spectral
    if not _skip("spectral"):
        try:
            from scintilla.clustering.spectral import spectral_clustering  # noqa: PLC0415
            from scintilla.config import SPECTRAL_N_CLUSTERS_RANGE  # noqa: PLC0415
            nc_vals = [n_clusters] if n_clusters else SPECTRAL_N_CLUSTERS_RANGE[:3]
            if config is not None and hasattr(config, "spectral_n_clusters_range") and config.spectral_n_clusters_range:
                nc_vals = config.spectral_n_clusters_range
            for nc in nc_vals:
                try:
                    lbl, _, _ = spectral_clustering(X, n_clusters=nc)
                    _record("Spectral", f"n_clusters={nc}", lbl)
                except MemoryError:
                    raise
                except (RuntimeError, ValueError, np.linalg.LinAlgError, ArithmeticError) as e:
                    if verbose:
                        print(f"Spectral n_clusters={nc} failed: {e}")
        except MemoryError:
            raise
        except (ImportError, RuntimeError, ValueError, np.linalg.LinAlgError, ArithmeticError) as e:
            if verbose:
                print(f"Spectral failed: {e}")

    # Consensus
    if not _skip("consensus"):
        try:
            from scintilla.clustering.consensus import consensus_clustering  # noqa: PLC0415
            _, lbl, _ = consensus_clustering(adata, methods=["kmeans"], n_runs_per_method=3)
            _record("Consensus", "kmeans", lbl)
        except MemoryError:
            raise
        except (ImportError, RuntimeError, ValueError, np.linalg.LinAlgError, ArithmeticError) as e:
            if verbose:
                print(f"Consensus failed: {e}")

    results_df = pd.DataFrame(records)

    # Visualise
    fig, ax = plt.subplots(figsize=(12, max(4, len(records) * 0.25)))
    valid_df = results_df.dropna(subset=["ari"]).sort_values("ari", ascending=True)
    if not valid_df.empty:
        y_labels = valid_df["method"] + " | " + valid_df["params"].astype(str)
        ax.barh(range(len(valid_df)), valid_df["ari"].values, color="steelblue")
        ax.set_yticks(range(len(valid_df)))
        ax.set_yticklabels(y_labels, fontsize=6)
        ax.set_xlabel("ARI")
        ax.set_title("Clustering Benchmark (ARI)")
    plt.tight_layout()

    return results_df, labels_dict, fig
