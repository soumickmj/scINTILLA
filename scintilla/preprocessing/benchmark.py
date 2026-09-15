"""Benchmarking of data transformations."""

from __future__ import annotations

import warnings
from typing import Callable, Dict, Optional, Tuple, Union

import anndata as ad
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.neighbors import NearestNeighbors

from scintilla.config import RANDOM_SEED, TRANSFORMATION_BENCHMARK_WEIGHTS
from scintilla.io.loaders import ensure_anndata
from scintilla.preprocessing.transformations import get_all_transformations


def _get_pca_matrix(
    X: np.ndarray, n_components: int = 20, random_state: int = RANDOM_SEED,
) -> np.ndarray:
    n_components = min(n_components, X.shape[0] - 1, X.shape[1])
    pca = PCA(n_components=n_components, random_state=random_state)
    return pca.fit_transform(X), pca


def _knn_overlap(X_orig: np.ndarray, X_trans: np.ndarray, k: int) -> float:
    """Fraction of k-nearest neighbours shared between original and transformed PCA spaces."""
    n = X_orig.shape[0]
    k = min(k, n - 1)
    nbrs_orig = NearestNeighbors(n_neighbors=k + 1, metric="euclidean").fit(X_orig)
    nbrs_trans = NearestNeighbors(n_neighbors=k + 1, metric="euclidean").fit(X_trans)
    idx_orig = nbrs_orig.kneighbors(return_distance=False)[:, 1:]
    idx_trans = nbrs_trans.kneighbors(return_distance=False)[:, 1:]
    overlaps = [len(np.intersect1d(idx_orig[i], idx_trans[i])) / k for i in range(n)]
    return float(np.mean(overlaps))


def _pca_preservation(
    X_orig: np.ndarray, X_trans: np.ndarray, n_components: int = 10,
    random_state: int = RANDOM_SEED,
) -> float:
    """Spearman correlation between top PC loadings before and after transformation."""
    nc = min(n_components, X_orig.shape[1] - 1, X_trans.shape[1] - 1, X_orig.shape[0] - 1)
    if nc < 1:
        return 0.0
    pca_orig = PCA(n_components=nc, random_state=random_state).fit(X_orig)
    pca_trans = PCA(n_components=nc, random_state=random_state).fit(X_trans)
    n_common = min(X_orig.shape[1], X_trans.shape[1])
    corrs = []
    for i in range(nc):
        lo = pca_orig.components_[i, :n_common]
        lt = pca_trans.components_[i, :n_common]
        r, _ = stats.spearmanr(np.abs(lo), np.abs(lt))
        corrs.append(abs(r) if not np.isnan(r) else 0.0)
    return float(np.mean(corrs))


def benchmark_transformations(
    data: Union[pd.DataFrame, ad.AnnData],
    transformations: Optional[Dict[str, Callable]] = None,
    weights: Optional[Dict[str, float]] = None,
    n_pca_components: Optional[int] = None,
    max_k: Optional[int] = None,
    cell_type_col: Optional[str] = None,
    scoring_method: Optional[str] = None,
    bootstrap_ci: Optional[bool] = None,
    n_bootstrap: Optional[int] = None,
    verbose: Optional[bool] = None,
    config=None,
    random_state: Optional[int] = None,
) -> Tuple[pd.DataFrame, Optional[str], Optional[ad.AnnData]]:
    """Benchmark multiple data transformations and return the best one.

    Parameters
    ----------
    cell_type_col:
        Column in ``adata.obs`` with cell-type labels.  When provided, the
        silhouette score is computed against these ground-truth labels
        instead of arbitrary KMeans k=3 clusters, giving a more meaningful
        assessment of how well the normalisation preserves biological
        structure.
    scoring_method:
        How to combine per-metric scores into a final ranking.

        - ``"weighted"`` (default) — weighted sum using *weights*.
        - ``"borda"`` — Borda-count rank aggregation.  Each metric
          independently ranks the transforms and the aggregate rank is
          computed without any weighting, eliminating the arbitrary weight
          vector entirely.
    bootstrap_ci:
        If ``True``, bootstrap-resample cells to compute 95 % confidence
        intervals for the composite score (columns ``composite_ci_low``,
        ``composite_ci_high``).  Uses *n_bootstrap* resamples.

        .. warning::

           Each bootstrap replicate re-applies every transformation and
           re-runs PCA and kNN, which can be extremely slow for large
           datasets.  For >10 000 cells consider sub-sampling first or
           running with ``verbose=True`` to monitor progress.
    n_bootstrap:
        Number of bootstrap resamples (default 200; kept low for
        tractability since each iteration re-runs PCA and kNN).

    Returns
    -------
    results_df : pd.DataFrame
        Per-transformation scores.
    best_name : str or None
        Name of the winning transformation, or ``None`` when all transforms
        fail.
    best_adata : ad.AnnData or None
        The input data transformed with the winning transformation, or
        ``None`` when all transforms fail.
    """
    if transformations is None:
        transformations = get_all_transformations()
    if weights is None:
        weights = TRANSFORMATION_BENCHMARK_WEIGHTS
    if n_pca_components is None:
        n_pca_components = getattr(config, "n_pca_comps", 20) if config is not None else 20
    if scoring_method is None:
        scoring_method = getattr(config, "scoring_method", "weighted") if config is not None else "weighted"
    if scoring_method not in {"weighted", "borda"}:
        raise ValueError(
            "Unknown transformation scoring_method "
            f"{scoring_method!r}; expected 'weighted' or 'borda'"
        )
    if bootstrap_ci is None:
        bootstrap_ci = getattr(config, "bootstrap_ci", False) if config is not None else False
    if n_bootstrap is None:
        n_bootstrap = getattr(config, "n_bootstrap", 200) if config is not None else 200
    if verbose is None:
        verbose = getattr(config, "verbose", True) if config is not None else True
    if random_state is None:
        random_state = getattr(config, "random_seed", RANDOM_SEED) if config is not None else RANDOM_SEED

    adata = ensure_anndata(data)
    X_raw = adata.X if not hasattr(adata.X, "toarray") else adata.X.toarray()
    X_raw = X_raw.astype(np.float64)

    # Resolve ground-truth labels for silhouette (if available)
    gt_labels = None
    if cell_type_col is not None and cell_type_col in adata.obs.columns:
        gt_labels = adata.obs[cell_type_col].values
        n_unique_labels = len(np.unique(gt_labels))
        if n_unique_labels < 2:
            gt_labels = None  # cannot compute silhouette with < 2 groups

    n = X_raw.shape[0]
    k = max_k if max_k is not None else min(15, max(2, n // 10))

    # PCA of raw data as reference
    n_pca = min(n_pca_components, n - 1, X_raw.shape[1])
    pca_raw = PCA(n_components=n_pca, random_state=random_state)
    X_pca_raw = pca_raw.fit_transform(X_raw)

    records = []
    for name, fn in transformations.items():
        if verbose:
            print(f"  Benchmarking: {name}")
        try:
            transform_kwargs = (
                {"random_state": random_state}
                if name == "glm_pca_transform" else {}
            )
            adata_t = fn(adata, **transform_kwargs)
            X_t = adata_t.X if not hasattr(adata_t.X, "toarray") else adata_t.X.toarray()
            X_t = X_t.astype(np.float64)
            # Replace NaN/inf
            X_t = np.nan_to_num(X_t, nan=0.0, posinf=0.0, neginf=0.0)

            n_pca_t = min(n_pca_components, n - 1, X_t.shape[1])
            pca_t = PCA(n_components=n_pca_t, random_state=random_state)
            X_pca_t = pca_t.fit_transform(X_t)

            # kNN overlap
            min_cols = min(X_pca_raw.shape[1], X_pca_t.shape[1])
            knn_ov = _knn_overlap(X_pca_raw[:, :min_cols], X_pca_t[:, :min_cols], k=k)

            # Silhouette: prefer ground-truth labels when available
            if gt_labels is not None:
                sil = float(silhouette_score(
                    X_pca_t, gt_labels, sample_size=min(1000, n),
                    random_state=random_state,
                ))
            else:
                k_sil = min(3, n - 1)
                if k_sil < 2:
                    sil = 0.0
                else:
                    km = KMeans(n_clusters=k_sil, random_state=random_state, n_init=5)
                    km_labels = km.fit_predict(X_pca_t)
                    if len(np.unique(km_labels)) < 2:
                        sil = 0.0
                    else:
                        sil = float(silhouette_score(
                            X_pca_t,
                            km_labels,
                            sample_size=min(1000, n),
                            random_state=random_state,
                        ))

            # Shapiro-Wilk fraction passing
            sw_pass = 0.0
            n_test = min(50, X_t.shape[1])
            rng_sw = np.random.default_rng(random_state)
            passed = 0
            shapiro_error = None
            for j in range(n_test):
                col = X_t[:, j]
                samp = col if len(col) <= 5000 else col[rng_sw.choice(len(col), 5000, replace=False)]
                try:
                    _, p = stats.shapiro(samp)
                    if p > 0.05:
                        passed += 1
                except ValueError as exc:
                    shapiro_error = shapiro_error or exc
            sw_pass = passed / max(n_test, 1)
            if shapiro_error is not None:
                warnings.warn(
                    f"{name} Shapiro-Wilk probe failed: {shapiro_error}",
                    stacklevel=2,
                )

            # Anderson-Darling fraction passing
            ad_pass = 0.0
            passed_ad = 0
            anderson_error = None
            for j in range(n_test):
                col = X_t[:, j]
                try:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore", FutureWarning)
                        res = stats.anderson(col, dist="norm")
                    if res.statistic < res.critical_values[2]:
                        passed_ad += 1
                except ValueError as exc:
                    anderson_error = anderson_error or exc
            ad_pass = passed_ad / max(n_test, 1)
            if anderson_error is not None:
                warnings.warn(
                    f"{name} Anderson-Darling probe failed: {anderson_error}",
                    stacklevel=2,
                )

            # PCA preservation
            n_common = min(X_raw.shape[1], X_t.shape[1])
            pca_pres = _pca_preservation(
                X_raw[:, :n_common], X_t[:, :n_common],
                random_state=random_state,
            )

            # Normalise silhouette to [0,1]
            sil_norm = np.clip((sil + 1) / 2, 0.0, 1.0)

            # Clamp all component scores to [0, 1] before weighting
            knn_ov = np.clip(knn_ov, 0.0, 1.0)
            pca_pres = np.clip(pca_pres, 0.0, 1.0)

            score = (
                weights.get("knn_overlap", 0.35) * knn_ov
                + weights.get("silhouette", 0.25) * sil_norm
                + weights.get("shapiro", 0.05) * sw_pass
                + weights.get("anderson", 0.05) * ad_pass
                + weights.get("pca_preservation", 0.30) * pca_pres
            )
            records.append({
                "transform": name,
                "status": "ok",
                "knn_overlap": knn_ov,
                "silhouette": sil,
                "shapiro_pass_frac": sw_pass,
                "anderson_pass_frac": ad_pass,
                "pca_preservation": pca_pres,
                "composite_score": score,
            })
        except MemoryError:
            raise
        except Exception as e:
            if verbose:
                print(f"    Failed: {e}")
            warnings.warn(f"{name} failed: {e}", stacklevel=2)
            records.append({
                "transform": name,
                "status": "failed",
                "knn_overlap": np.nan,
                "silhouette": np.nan,
                "shapiro_pass_frac": np.nan,
                "anderson_pass_frac": np.nan,
                "pca_preservation": np.nan,
                "composite_score": np.nan,
                "failure_reason": str(e),
            })

    results_df = pd.DataFrame(records).sort_values("composite_score", ascending=False)
    valid_rows = results_df.dropna(subset=["composite_score"])
    if valid_rows.empty:
        return results_df, None, None

    # --- Borda-count rank aggregation (alternative to weighted sum) ---
    if scoring_method == "borda":
        from scintilla.statistical_tests.rank_aggregation import borda_count  # noqa: PLC0415

        metric_cols = ["knn_overlap", "silhouette", "shapiro_pass_frac",
                       "anderson_pass_frac", "pca_preservation"]
        valid_for_borda = results_df.dropna(subset=metric_cols)
        if not valid_for_borda.empty:
            score_sub = valid_for_borda.set_index("transform")[metric_cols]
            borda_df = borda_count(score_sub, higher_is_better=[True] * len(metric_cols))
            # Merge borda_rank back
            results_df = results_df.merge(
                borda_df[["borda_score", "borda_rank"]].reset_index(),
                on="transform", how="left",
            )
            results_df = results_df.sort_values("borda_rank", na_position="last")
        else:
            results_df["borda_score"] = np.nan
            results_df["borda_rank"] = np.nan

    # --- Bootstrap confidence intervals for composite score ---
    if bootstrap_ci and not valid_rows.empty:
        import warnings as _bw  # noqa: PLC0415
        if n > 10_000:
            _bw.warn(
                f"benchmark_transformations: bootstrap_ci=True with {n:,} cells.  "
                f"Each of the {n_bootstrap} resamples re-applies every "
                f"transformation and re-runs PCA/kNN, which will be very "
                f"slow.  Consider sub-sampling to ≤10 000 cells first.",
                stacklevel=2,
            )
        from scintilla.statistical_tests.bootstrap import (  # noqa: PLC0415
            bootstrap_resample_metrics,
        )
        ci_low_list, ci_high_list = [], []
        rng_boot = np.random.default_rng(random_state)
        for _, row in results_df.iterrows():
            tname = row["transform"]
            if pd.isna(row.get("composite_score")):
                ci_low_list.append(np.nan)
                ci_high_list.append(np.nan)
                continue
            try:
                fn = transformations[tname]
                boot_scores = []
                bootstrap_error = None
                for b in range(n_bootstrap):
                    idx = rng_boot.integers(0, n, size=n)
                    adata_sub = adata[idx].copy()
                    try:
                        transform_kwargs = (
                            {"random_state": random_state}
                            if tname == "glm_pca_transform" else {}
                        )
                        adata_bt = fn(adata_sub, **transform_kwargs)
                        X_bt = adata_bt.X if not hasattr(adata_bt.X, "toarray") else adata_bt.X.toarray()
                        X_bt = np.nan_to_num(X_bt.astype(np.float64))
                        n_pca_bt = min(n_pca_components, n - 1, X_bt.shape[1])
                        X_pca_bt = PCA(
                            n_components=n_pca_bt, random_state=random_state,
                        ).fit_transform(X_bt)
                        # Quick composite: kNN overlap + silhouette
                        min_c = min(X_pca_raw.shape[1], X_pca_bt.shape[1])
                        knn_ = _knn_overlap(
                            X_pca_raw[idx, :min_c], X_pca_bt[:, :min_c], k=k,
                        )
                        if gt_labels is not None:
                            sil_ = float(silhouette_score(
                                X_pca_bt, gt_labels[idx],
                                sample_size=min(500, n),
                                random_state=random_state,
                            ))
                        else:
                            sil_ = 0.0
                        sil_n = np.clip((sil_ + 1) / 2, 0, 1)
                        sc_ = (
                            weights.get("knn_overlap", 0.35) * np.clip(knn_, 0, 1)
                            + weights.get("silhouette", 0.25) * sil_n
                        )
                        boot_scores.append(sc_)
                    except (RuntimeError, ValueError, np.linalg.LinAlgError, ArithmeticError) as exc:
                        bootstrap_error = bootstrap_error or exc
                if bootstrap_error is not None:
                    warnings.warn(
                        f"{tname} bootstrap probe failed: {bootstrap_error}",
                        stacklevel=2,
                    )
                if boot_scores:
                    ci = bootstrap_resample_metrics(
                        np.array(boot_scores), B=min(500, len(boot_scores)),
                        seed=random_state,
                    )
                    ci_low_list.append(ci["ci_low"])
                    ci_high_list.append(ci["ci_high"])
                else:
                    ci_low_list.append(np.nan)
                    ci_high_list.append(np.nan)
            except (KeyError, RuntimeError, ValueError, np.linalg.LinAlgError, ArithmeticError) as exc:
                warnings.warn(
                    f"{tname} bootstrap confidence interval failed: {exc}",
                    stacklevel=2,
                )
                ci_low_list.append(np.nan)
                ci_high_list.append(np.nan)
        results_df["composite_ci_low"] = ci_low_list
        results_df["composite_ci_high"] = ci_high_list

    # --- Select best ---
    sort_col = "borda_rank" if scoring_method == "borda" and "borda_rank" in results_df.columns else "composite_score"
    ascending = sort_col == "borda_rank"
    results_df = results_df.sort_values(sort_col, ascending=ascending, na_position="last")
    best_name = results_df.iloc[0]["transform"]
    best_func = transformations[best_name]
    best_kwargs = (
        {"random_state": random_state}
        if best_name == "glm_pca_transform" else {}
    )
    best_adata = best_func(adata, **best_kwargs)
    return results_df, best_name, best_adata
