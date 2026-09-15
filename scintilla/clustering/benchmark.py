"""Comprehensive clustering benchmark."""

from __future__ import annotations

import warnings
from typing import Dict, Optional, Tuple, Union

import anndata as ad
import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_mutual_info_score, adjusted_rand_score

import matplotlib.pyplot as plt

from tqdm.auto import tqdm

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
    n_jobs: int = 1,
    verbose: Optional[bool] = None,
    config=None,
    random_state: Optional[int] = None,
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

    if random_state is None:
        random_state = getattr(config, "random_seed", RANDOM_SEED) if config is not None else RANDOM_SEED
    if verbose is None:
        verbose = getattr(config, "verbose", True) if config is not None else True

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

    def _score(method, params, labels):
        n_found = len(np.unique(labels[labels != -1]))
        valid = labels[labels != -1]
        true_valid = true_labels[labels != -1]
        noise_frac = np.mean(labels == -1) if len(labels) > 0 else 0.0
        if noise_frac > 0.5:
            ari = np.nan
            ami = np.nan
        else:
            ari = adjusted_rand_score(true_valid, valid) if len(valid) > 0 else np.nan
            ami = adjusted_mutual_info_score(true_valid, valid) if len(valid) > 0 else np.nan
        return {
            "method": method, "params": params, "ari": ari, "ami": ami,
            "n_clusters": n_found, "noise_fraction": round(noise_frac, 4),
            "status": "ok",
        }, labels

    def _failure(method, params, exc):
        warnings.warn(f"{method} {params} failed: {exc}", stacklevel=3)
        return {
            "method": method,
            "params": params,
            "status": "failed",
            "failure_reason": str(exc),
            "ari": np.nan,
            "ami": np.nan,
            "n_clusters": np.nan,
            "noise_fraction": np.nan,
        }

    def _skipped(method, reason):
        """Row for a method the environment cannot offer at all.

        An uninstalled optional backend is not a failure of the method: it
        never ran.  Keeping it distinct from status="failed" means a genuine
        error is not hidden among missing-dependency noise, and no warning is
        emitted for a choice the user already made at install time.
        """
        return {
            "method": method,
            "params": "unavailable",
            "status": "skipped",
            "failure_reason": reason,
            "ari": np.nan,
            "ami": np.nan,
            "n_clusters": np.nan,
            "noise_fraction": np.nan,
        }

    # ── Define each method group as a callable ──────────────────────
    def _run_kmeans():
        recs, lbls = [], {}
        for init in ["k-means++", "random"]:
            try:
                lbl, _, _ = kmeans_clustering(
                    X, n_clusters, init=init, random_state=random_state,
                )
                rec, lbl = _score("KMeans", f"init={init}", lbl)
                recs.append(rec); lbls[f"KMeans_init={init}"] = lbl
            except MemoryError:
                raise
            except (RuntimeError, ValueError, np.linalg.LinAlgError, ArithmeticError) as exc:
                recs.append(_failure("KMeans", f"init={init}", exc))
        try:
            lbl, _, _ = kmeans_clustering(
                X, n_clusters, spherical=True, random_state=random_state,
            )
            rec, lbl = _score("KMeans_spherical", "spherical=True", lbl)
            recs.append(rec); lbls["KMeans_spherical_spherical=True"] = lbl
        except MemoryError:
            raise
        except (RuntimeError, ValueError, np.linalg.LinAlgError, ArithmeticError) as exc:
            recs.append(_failure("KMeans_spherical", "spherical=True", exc))
        try:
            lbl, _, _ = kmeans_clustering(
                X, n_clusters, bisecting=True, random_state=random_state,
            )
            rec, lbl = _score("KMeans_bisecting", "bisecting=True", lbl)
            recs.append(rec); lbls["KMeans_bisecting_bisecting=True"] = lbl
        except MemoryError:
            raise
        except (RuntimeError, ValueError, np.linalg.LinAlgError, ArithmeticError) as exc:
            recs.append(_failure("KMeans_bisecting", "bisecting=True", exc))
        return recs, lbls

    def _run_hierarchical():
        recs, lbls = [], {}
        for metric in DISTANCE_METRICS:
            for lnk in LINKAGE_METHODS:
                if lnk == "ward" and metric != "euclidean":
                    recs.append({
                        "method": "Hierarchical",
                        "params": f"{metric}/{lnk}",
                        "status": "skipped",
                        "failure_reason": "ward linkage requires euclidean distance",
                        "ari": np.nan,
                        "ami": np.nan,
                        "n_clusters": np.nan,
                        "noise_fraction": np.nan,
                    })
                    continue
                try:
                    lbl = hierarchical_sklearn(X, n_clusters, metric=metric, linkage_method=lnk)
                    rec, lbl = _score("Hierarchical", f"{metric}/{lnk}", lbl)
                    recs.append(rec); lbls[f"Hierarchical_{metric}/{lnk}"] = lbl
                except MemoryError:
                    raise
                except (RuntimeError, ValueError, np.linalg.LinAlgError, ArithmeticError) as exc:
                    recs.append(_failure("Hierarchical", f"{metric}/{lnk}", exc))
        return recs, lbls

    def _run_dbscan():
        recs, lbls = [], {}
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
                    rec, lbl = _score("DBSCAN", f"eps={eps},min_samples={minsamp}", lbl)
                    recs.append(rec); lbls[f"DBSCAN_eps={eps},min_samples={minsamp}"] = lbl
                except MemoryError:
                    raise
                except (RuntimeError, ValueError, np.linalg.LinAlgError, ArithmeticError) as exc:
                    recs.append(_failure(
                        "DBSCAN", f"eps={eps},min_samples={minsamp}", exc,
                    ))
        return recs, lbls

    def _run_leiden():
        recs, lbls = [], {}
        try:
            from scintilla.clustering.leiden import leiden_clustering  # noqa: PLC0415
            _leiden_res = LEIDEN_RESOLUTIONS
            if config is not None and hasattr(config, "leiden_resolutions") and config.leiden_resolutions:
                _leiden_res = config.leiden_resolutions

            if adaptive_resolution:
                from scintilla.statistical_tests.adaptive import adaptive_resolution_search  # noqa: PLC0415
                def _rl(res):
                    return leiden_clustering(
                        adata, resolution=res, use_rep=use_rep,
                        random_state=random_state,
                    )
                def _ml(labels):
                    valid = labels[labels != -1]
                    tv = true_labels[labels != -1]
                    return adjusted_rand_score(tv, valid) if len(valid) > 0 else 0.0
                search_result = adaptive_resolution_search(_rl, _ml, _leiden_res, refine_steps=5)
                for res, score in search_result["all_scores"]:
                    lbl = _rl(res)
                    rec, lbl = _score("Leiden", f"resolution={res:.4f}", lbl)
                    recs.append(rec); lbls[f"Leiden_resolution={res:.4f}"] = lbl
            elif resolution_selection == "nvi_stability":
                from scintilla.statistical_tests.adaptive import nvi_stability  # noqa: PLC0415
                for res in _leiden_res:
                    try:
                        lbl = leiden_clustering(
                            adata, resolution=res, use_rep=use_rep,
                            random_state=random_state,
                        )
                        rec, lbl = _score("Leiden", f"resolution={res}", lbl)
                        recs.append(rec); lbls[f"Leiden_resolution={res}"] = lbl
                    except MemoryError:
                        raise
                    except Exception as exc:
                        warnings.warn(f"Leiden failed: {exc}", stacklevel=2)
                        recs.append({
                            "method": "Leiden", "params": f"resolution={res}",
                            "status": "failed", "failure_reason": str(exc),
                            "ari": np.nan, "ami": np.nan,
                            "n_clusters": np.nan, "noise_fraction": np.nan,
                        })
            else:
                for res in _leiden_res:
                    try:
                        lbl = leiden_clustering(
                            adata, resolution=res, use_rep=use_rep,
                            random_state=random_state,
                        )
                        rec, lbl = _score("Leiden", f"resolution={res}", lbl)
                        recs.append(rec); lbls[f"Leiden_resolution={res}"] = lbl
                    except MemoryError:
                        raise
                    except (RuntimeError, ValueError, np.linalg.LinAlgError, ArithmeticError) as exc:
                        recs.append(_failure("Leiden", f"resolution={res}", exc))
        except ImportError as exc:
            recs.append(_skipped("Leiden", str(exc)))
        return recs, lbls

    def _run_hdbscan():
        recs, lbls = [], {}
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
            _, lbl = hdbscan_clustering(X, min_cluster_size_range=_hdb_sizes, min_samples_range=_hdb_samples)
            rec, lbl = _score("HDBSCAN", "best", lbl)
            recs.append(rec); lbls["HDBSCAN_best"] = lbl
        except MemoryError:
            raise
        except ImportError as exc:
            recs.append(_skipped("HDBSCAN", str(exc)))
        except (RuntimeError, ValueError, np.linalg.LinAlgError, ArithmeticError) as exc:
            recs.append(_failure("HDBSCAN", "best", exc))
        return recs, lbls

    def _run_louvain():
        recs, lbls = [], {}
        try:
            from scintilla.clustering.louvain import louvain_clustering  # noqa: PLC0415
            from scintilla.config import LOUVAIN_RESOLUTIONS  # noqa: PLC0415
            _louv_res = LOUVAIN_RESOLUTIONS
            if config is not None and hasattr(config, "louvain_resolutions") and config.louvain_resolutions:
                _louv_res = config.louvain_resolutions
            for res in _louv_res:
                try:
                    lbl = louvain_clustering(
                        adata, resolution=res, use_rep=use_rep,
                        random_state=random_state,
                    )
                    rec, lbl = _score("Louvain", f"resolution={res}", lbl)
                    recs.append(rec); lbls[f"Louvain_resolution={res}"] = lbl
                except MemoryError:
                    raise
                except (RuntimeError, ValueError, np.linalg.LinAlgError, ArithmeticError) as exc:
                    recs.append(_failure("Louvain", f"resolution={res}", exc))
        except ImportError as exc:
            recs.append(_skipped("Louvain", str(exc)))
        return recs, lbls

    def _run_spectral():
        recs, lbls = [], {}
        try:
            from scintilla.clustering.spectral import spectral_clustering  # noqa: PLC0415
            from scintilla.config import SPECTRAL_N_CLUSTERS_RANGE  # noqa: PLC0415
            nc_vals = [n_clusters] if n_clusters else SPECTRAL_N_CLUSTERS_RANGE[:3]
            if config is not None and hasattr(config, "spectral_n_clusters_range") and config.spectral_n_clusters_range:
                nc_vals = config.spectral_n_clusters_range
            for nc in nc_vals:
                try:
                    lbl, _, _ = spectral_clustering(
                        X, n_clusters=nc, random_state=random_state,
                    )
                    rec, lbl = _score("Spectral", f"n_clusters={nc}", lbl)
                    recs.append(rec); lbls[f"Spectral_n_clusters={nc}"] = lbl
                except MemoryError:
                    raise
                except (RuntimeError, ValueError, np.linalg.LinAlgError, ArithmeticError) as exc:
                    recs.append(_failure("Spectral", f"n_clusters={nc}", exc))
        except MemoryError:
            raise
        except ImportError as exc:
            recs.append(_skipped("Spectral", str(exc)))
        except (RuntimeError, ValueError, np.linalg.LinAlgError, ArithmeticError) as exc:
            recs.append(_failure("Spectral", "unavailable", exc))
        return recs, lbls

    def _run_consensus():
        recs, lbls = [], {}
        try:
            from scintilla.clustering.consensus import consensus_clustering  # noqa: PLC0415
            _, lbl, _ = consensus_clustering(
                adata, methods=["kmeans"], n_runs_per_method=3,
                random_state=random_state,
            )
            rec, lbl = _score("Consensus", "kmeans", lbl)
            recs.append(rec); lbls["Consensus_kmeans"] = lbl
        except MemoryError:
            raise
        except ImportError as exc:
            recs.append(_skipped("Consensus", str(exc)))
        except (RuntimeError, ValueError, np.linalg.LinAlgError, ArithmeticError) as exc:
            recs.append(_failure("Consensus", "kmeans", exc))
        return recs, lbls

    # ── Build task list and execute ─────────────────────────────────
    _method_runners = {
        "kmeans": _run_kmeans,
        "hierarchical": _run_hierarchical,
        "dbscan": _run_dbscan,
        "leiden": _run_leiden,
        "hdbscan": _run_hdbscan,
        "louvain": _run_louvain,
        "spectral": _run_spectral,
        "consensus": _run_consensus,
    }
    _methods_to_run = {k: v for k, v in _method_runners.items() if not _skip(k)}

    if n_jobs == 1:
        pbar = tqdm(total=len(_methods_to_run), desc="Clustering benchmark",
                    disable=not verbose)
        for name, runner in _methods_to_run.items():
            pbar.set_postfix_str(name)
            recs, lbls = runner()
            records.extend(recs)
            labels_dict.update(lbls)
            pbar.update(1)
        pbar.set_postfix_str("done")
        pbar.close()
    else:
        from joblib import Parallel, delayed  # noqa: PLC0415
        if verbose:
            print(f"Running {len(_methods_to_run)} clustering methods in parallel (n_jobs={n_jobs})...")
        results = Parallel(n_jobs=n_jobs, prefer="threads")(
            delayed(runner)() for runner in _methods_to_run.values()
        )
        for recs, lbls in results:
            records.extend(recs)
            labels_dict.update(lbls)

    results_df = pd.DataFrame(records)

    # Visualise
    valid_df = results_df.dropna(subset=["ari"]).sort_values("ari", ascending=True)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, max(4, len(records) * 0.25)))
    if not valid_df.empty:
        y_labels = valid_df["method"] + " | " + valid_df["params"].astype(str)
        y_pos = range(len(valid_df))
        ax1.barh(y_pos, valid_df["ari"].values, color="steelblue")
        ax1.set_yticks(y_pos)
        ax1.set_yticklabels(y_labels, fontsize=6)
        ax1.set_xlabel("ARI")
        ax1.set_title("Clustering Benchmark (ARI)")

        ami_sorted = valid_df.sort_values("ami", ascending=True)
        y_labels_ami = ami_sorted["method"] + " | " + ami_sorted["params"].astype(str)
        y_pos_ami = range(len(ami_sorted))
        ax2.barh(y_pos_ami, ami_sorted["ami"].values, color="darkorange")
        ax2.set_yticks(y_pos_ami)
        ax2.set_yticklabels(y_labels_ami, fontsize=6)
        ax2.set_xlabel("AMI")
        ax2.set_title("Clustering Benchmark (AMI)")
    plt.tight_layout()

    return results_df, labels_dict, fig
