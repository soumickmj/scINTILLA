"""Comprehensive classification benchmark across models and feature spaces."""

from __future__ import annotations

import warnings
from typing import Optional, Tuple, Union

import anndata as ad
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.decomposition import PCA

import matplotlib.pyplot as plt

from scintilla.config import RANDOM_SEED, DEFAULT_TEST_SIZE, DEFAULT_CV_FOLDS
from scintilla.io.loaders import ensure_anndata
from scintilla.classification.models import (
    lda_classification,
    qda_classification,
    svm_classification,
    random_forest_classification,
    logistic_regression_classification,
    mlp_classification,
    knn_classification,
    gradient_boosting_classification,
    naive_bayes_classification,
    stacking_ensemble_classification,
)


_MODEL_FNS = {
    "LogReg": logistic_regression_classification,
    "RF": random_forest_classification,
    "SVM": svm_classification,
    "MLP": mlp_classification,
    "LDA": lda_classification,
    "QDA": qda_classification,
    "kNN": knn_classification,
    "GradientBoosting": gradient_boosting_classification,
    "NaiveBayes": naive_bayes_classification,
    "StackingEnsemble": stacking_ensemble_classification,
}

_STOCHASTIC_MODEL_NAMES = {
    "LogReg", "RF", "SVM", "MLP", "XGBoost", "LightGBM",
    "GradientBoosting", "StackingEnsemble",
}
_OMITTED = object()

# Optional models (xgboost, lightgbm).  Their wrappers live in models.py and
# import their backend lazily, so importing the wrapper never fails.  Probe the
# backend itself instead: an absent package means the model is simply not
# offered, which keeps "not installed" distinct from a real failure inside the
# wrapper (which still surfaces as status="failed").
_OPTIONAL_MODEL_BACKENDS = {
    "XGBoost": ("xgboost", "xgboost_classification"),
    "LightGBM": ("lightgbm", "lightgbm_classification"),
}


def _optional_backend_available(module_name: str) -> bool:
    """Return True when *module_name* can be imported without importing it."""
    import importlib.util  # noqa: PLC0415

    try:
        return importlib.util.find_spec(module_name) is not None
    except (ImportError, ValueError):
        return False


def _resolve_optional_models() -> dict:
    """Return the optional model wrappers whose backend is installed."""
    import importlib  # noqa: PLC0415

    resolved = {}
    models = importlib.import_module("scintilla.classification.models")
    for name, (backend, attr) in _OPTIONAL_MODEL_BACKENDS.items():
        if _optional_backend_available(backend):
            resolved[name] = getattr(models, attr)
    return resolved


_MODEL_FNS.update(_resolve_optional_models())


def _benchmark_cv(
    X: np.ndarray,
    y: np.ndarray,
    n_folds: int,
    use_pca: bool,
    n_pca_comps: int,
    verbose: bool,
    random_state: int,
) -> Tuple[pd.DataFrame, plt.Figure]:
    """Stratified k-fold cross-validation benchmark."""
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)
    # Collect full metrics dict per fold for each (space, model)
    from collections import defaultdict
    fold_results: dict = defaultdict(list)
    fold_failures: dict = defaultdict(list)

    for fold_idx, (tr_idx, te_idx) in enumerate(skf.split(X, y)):
        X_tr_full, X_te_full = X[tr_idx], X[te_idx]
        y_tr, y_te = y[tr_idx], y[te_idx]

        spaces = {"Gene": (X_tr_full, X_te_full)}
        if use_pca:
            n_c = min(n_pca_comps, X_tr_full.shape[1] - 1, X_tr_full.shape[0] - 1)
            pca = PCA(n_components=n_c, random_state=random_state)
            X_tr_pca = pca.fit_transform(X_tr_full)
            X_te_pca = pca.transform(X_te_full)
            spaces["PCA"] = (X_tr_pca, X_te_pca)

        for space_name, (X_tr, X_te) in spaces.items():
            for model_name, fn in _MODEL_FNS.items():
                if verbose:
                    print(f"  Fold {fold_idx+1}/{n_folds} | {space_name} | {model_name}")
                key = (space_name, model_name)
                try:
                    kwargs = {"random_state": random_state} if model_name in _STOCHASTIC_MODEL_NAMES else {}
                    _, metrics = fn(X_tr, X_te, y_tr, y_te, **kwargs)
                    fold_results[key].append(metrics)
                except MemoryError:
                    raise
                except Exception as e:
                    if verbose:
                        print(f"    Failed: {e}")
                    fold_results[key].append(None)
                    fold_failures[key].append(str(e))

    records = []
    for (space_name, model_name), metrics_list in fold_results.items():
        key = (space_name, model_name)
        failures = fold_failures[key]
        # Collect all numeric metric keys across successful folds
        valid_metrics = [m for m in metrics_list if m is not None]
        row = {
            "space": space_name,
            "model": model_name,
            "status": "failed" if not valid_metrics else "ok",
            "failure_reason": "; ".join(failures) if failures else None,
        }
        if valid_metrics:
            all_keys = {k for m in valid_metrics for k, v in m.items()
                        if isinstance(v, (int, float)) and k != "confusion_matrix"}
            for metric_key in sorted(all_keys):
                values = np.array(
                    [m.get(metric_key, np.nan) for m in valid_metrics],
                    dtype=np.float64,
                )
                row[f"mean_{metric_key}"] = float(np.nanmean(values))
                row[f"std_{metric_key}"] = float(np.nanstd(values, ddof=1))
                n_valid = int(np.sum(np.isfinite(values)))
                # SE of the mean across folds — correct uncertainty on the
                # mean estimate.  SD / sqrt(k) is ~2.2× smaller than SD
                # for k=5, which matters for comparing models visually.
                row[f"se_{metric_key}"] = float(
                    np.nanstd(values, ddof=1) / np.sqrt(max(n_valid, 1))
                )
            # Keep top-level 'accuracy' as alias for backward compatibility
            row["accuracy"] = row.get("mean_accuracy", np.nan)
        else:
            row["mean_accuracy"] = np.nan
            row["std_accuracy"] = np.nan
            row["se_accuracy"] = np.nan
            row["accuracy"] = np.nan
        records.append(row)
        if failures:
            warnings.warn(
                f"{model_name} failed: {failures[0]}", stacklevel=2,
            )

    results_df = pd.DataFrame(records)
    fig, ax = plt.subplots(figsize=(12, 5))
    if "mean_accuracy" in results_df.columns:
        pivot = results_df.pivot_table(index="model", columns="space", values="mean_accuracy")
        # Use standard error of the mean (SE = SD/√k) for error bars so
        # they represent uncertainty on the mean estimate, not between-fold
        # variability.  SE is the statistically correct quantity here.
        if "se_accuracy" in results_df.columns:
            yerr_pivot = results_df.pivot_table(index="model", columns="space", values="se_accuracy")
            yerr_label = "± SE"
        else:
            yerr_pivot = results_df.pivot_table(index="model", columns="space", values="std_accuracy")
            yerr_label = "± SD"
        pivot.plot(kind="bar", ax=ax, yerr=yerr_pivot)
        ax.set_title(f"Classification Benchmark ({n_folds}-fold CV, mean accuracy {yerr_label})")
        ax.set_ylabel("Accuracy")
        ax.set_xlabel("Model")
        plt.xticks(rotation=45)
        plt.tight_layout()
    return results_df, fig


def benchmark_models_comprehensive(
    data: Union[pd.DataFrame, ad.AnnData],
    target_col: str,
    test_size: Optional[float] = None,
    use_pca: bool = True,
    n_pca_comps: Optional[int] = None,
    verbose: Optional[bool] = None,
    cv_folds: Optional[int] = _OMITTED,
    bootstrap_ci: Optional[bool] = None,
    n_bootstrap: Optional[int] = None,
    estimator: Optional[str] = None,
    no_info_method: Optional[str] = None,
    config=None,
    random_state: Optional[int] = None,
) -> Tuple[pd.DataFrame, plt.Figure]:
    """Benchmark all classifiers in gene space and PCA space.

    Parameters
    ----------
    data:
        Input data.
    target_col:
        Column in obs with class labels.
    test_size:
        Fraction of data used for testing (single-split mode).
    use_pca:
        Also evaluate models in PCA space.
    n_pca_comps:
        Number of PCA components.
    verbose:
        Print progress.
    cv_folds:
        If not None, run stratified k-fold cross-validation instead of a
        single train/test split.  Results will include ``mean_accuracy``
        and ``std_accuracy`` columns.  Defaults to 5.
    bootstrap_ci:
        If True, compute BCa bootstrap confidence intervals for each
        metric in the single-split / .632+ paths.
    n_bootstrap:
        Number of bootstrap replicates for CIs.
    estimator:
        Evaluation strategy.  ``"cv"`` (default) uses stratified k-fold
        cross-validation (honours *cv_folds*).  ``"holdout"`` forces a
        single train/test split.  ``"bootstrap_632plus"`` uses the .632+
        bootstrap estimator from Efron & Tibshirani (1997).
    no_info_method:
        How to estimate the no-information rate for the .632+ estimator.
        ``"analytical"`` (default) uses ``sum(p_k^2)``, which is exact
        for accuracy but only approximate for other metrics.
        ``"permutation"`` averages over random label permutations and is
        correct for any metric, at the cost of extra computation.

    Returns
    -------
    results_df : pd.DataFrame
    fig : matplotlib Figure
    """
    if test_size is None:
        test_size = getattr(config, "test_size", DEFAULT_TEST_SIZE) if config is not None else DEFAULT_TEST_SIZE
    if n_pca_comps is None:
        n_pca_comps = getattr(config, "n_pca_comps", 30) if config is not None else 30
    if verbose is None:
        verbose = getattr(config, "verbose", True) if config is not None else True
    if cv_folds is _OMITTED:
        cv_folds = getattr(config, "cv_folds", DEFAULT_CV_FOLDS) if config is not None else DEFAULT_CV_FOLDS
    if bootstrap_ci is None:
        bootstrap_ci = getattr(config, "bootstrap_ci", False) if config is not None else False
    if n_bootstrap is None:
        n_bootstrap = getattr(config, "n_bootstrap", 2000) if config is not None else 2000
    if estimator is None:
        estimator = getattr(config, "classification_estimator", "cv") if config is not None else "cv"
    if no_info_method is None:
        no_info_method = getattr(config, "no_info_method", "analytical") if config is not None else "analytical"
    if random_state is None:
        random_state = getattr(config, "random_seed", RANDOM_SEED) if config is not None else RANDOM_SEED

    adata = ensure_anndata(data, target_col=target_col)
    if target_col not in adata.obs.columns:
        raise KeyError(f"Column '{target_col}' not found in obs.")

    X_full = adata.X if not hasattr(adata.X, "toarray") else adata.X.toarray()
    X_full = X_full.astype(np.float64)
    y = adata.obs[target_col].values

    # ------------------------------------------------------------------
    # Estimator dispatch
    # ------------------------------------------------------------------
    if estimator == "cv" and cv_folds is not None and cv_folds >= 2:
        return _benchmark_cv(
            X_full, y, cv_folds, use_pca, n_pca_comps, verbose, random_state,
        )

    if estimator == "bootstrap_632plus":
        return _benchmark_632plus(
            X_full, y, use_pca, n_pca_comps, verbose,
            B=n_bootstrap, bootstrap_ci=bootstrap_ci,
            no_info_method=no_info_method,
            random_state=random_state,
        )

    # ------------------------------------------------------------------
    # Single holdout split (default when estimator="holdout")
    # ------------------------------------------------------------------
    X_tr_full, X_te_full, y_tr, y_te = train_test_split(
        X_full, y, test_size=test_size, random_state=random_state, stratify=y
    )

    spaces = {"Gene": (X_tr_full, X_te_full)}

    if use_pca:
        n_c = min(n_pca_comps, X_tr_full.shape[1] - 1, X_tr_full.shape[0] - 1)
        pca = PCA(n_components=n_c, random_state=random_state)
        X_tr_pca = pca.fit_transform(X_tr_full)
        X_te_pca = pca.transform(X_te_full)
        spaces["PCA"] = (X_tr_pca, X_te_pca)

    records = []
    for space_name, (X_tr, X_te) in spaces.items():
        for model_name, fn in _MODEL_FNS.items():
            if verbose:
                print(f"  {space_name} | {model_name}")
            try:
                kwargs = {"random_state": random_state} if model_name in _STOCHASTIC_MODEL_NAMES else {}
                model, metrics = fn(X_tr, X_te, y_tr, y_te, **kwargs)
                row = {
                    "space": space_name, "model": model_name,
                    "status": "ok", "failure_reason": None,
                }
                row.update(metrics)

                # BCa bootstrap CIs on the holdout test set
                if bootstrap_ci:
                    y_pred = model.predict(X_te)
                    _add_bootstrap_cis(
                        row, y_te, y_pred, n_bootstrap, seed=random_state,
                    )

                records.append(row)
            except MemoryError:
                raise
            except Exception as e:
                if verbose:
                    print(f"    Failed: {e}")
                warnings.warn(f"{model_name} failed: {e}", stacklevel=2)
                records.append({
                    "space": space_name, "model": model_name,
                    "status": "failed", "accuracy": np.nan,
                    "failure_reason": str(e),
                })

    results_df = pd.DataFrame(records)

    # Figure: accuracy grouped bar
    fig, ax = plt.subplots(figsize=(12, 5))
    if "accuracy" in results_df.columns and results_df["accuracy"].notna().any():
        pivot = results_df.pivot_table(index="model", columns="space", values="accuracy")
        pivot.plot(kind="bar", ax=ax)
        ax.set_title("Classification Benchmark (Accuracy)")
        ax.set_ylabel("Accuracy")
        ax.set_xlabel("Model")
        plt.xticks(rotation=45)
        plt.tight_layout()

    return results_df, fig


# ------------------------------------------------------------------
# Helper: BCa bootstrap CIs for classification metrics
# ------------------------------------------------------------------

_CI_METRICS = {
    "accuracy": lambda yt, yp: float(np.mean(yt == yp)),
    "f1": None,  # populated lazily
}


def _add_bootstrap_cis(
    row: dict,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    B: int,
    seed: int = RANDOM_SEED,
) -> None:
    """Mutate *row* in-place, adding ``<metric>_ci_low/high`` columns."""
    from sklearn.metrics import accuracy_score, f1_score  # noqa: PLC0415
    from scintilla.statistical_tests.bootstrap import bootstrap_metric_ci  # noqa: PLC0415

    metric_fns = {
        "accuracy": accuracy_score,
        "f1": lambda yt, yp: f1_score(yt, yp, average="macro", zero_division=0),
    }
    for name, fn in metric_fns.items():
        try:
            ci = bootstrap_metric_ci(y_true, y_pred, fn, B=B, seed=seed)
            row[f"{name}_ci_low"] = ci["ci_low"]
            row[f"{name}_ci_high"] = ci["ci_high"]
        except Exception:
            row[f"{name}_ci_low"] = np.nan
            row[f"{name}_ci_high"] = np.nan


# ------------------------------------------------------------------
# .632+ bootstrap estimator path
# ------------------------------------------------------------------

# Map model names → unfitted sklearn estimator constructors
def _get_model_instance(model_name: str, random_state: int = RANDOM_SEED):
    """Return an unfitted estimator instance for *model_name*."""
    from sklearn.discriminant_analysis import (  # noqa: PLC0415
        LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis,
    )
    from sklearn.ensemble import (  # noqa: PLC0415
        GradientBoostingClassifier, RandomForestClassifier,
    )
    from sklearn.linear_model import LogisticRegression  # noqa: PLC0415
    from sklearn.naive_bayes import GaussianNB  # noqa: PLC0415
    from sklearn.neighbors import KNeighborsClassifier  # noqa: PLC0415
    from sklearn.neural_network import MLPClassifier  # noqa: PLC0415
    from sklearn.svm import SVC  # noqa: PLC0415

    _map = {
        "LogReg": lambda: LogisticRegression(max_iter=500, random_state=random_state, solver="lbfgs"),
        "RF": lambda: RandomForestClassifier(n_estimators=100, random_state=random_state),
        "SVM": lambda: SVC(kernel="rbf", probability=True, random_state=random_state),
        "MLP": lambda: MLPClassifier(max_iter=500, random_state=random_state, hidden_layer_sizes=(100, 50)),
        "LDA": lambda: LinearDiscriminantAnalysis(),
        "QDA": lambda: QuadraticDiscriminantAnalysis(reg_param=0.01),
        "kNN": lambda: KNeighborsClassifier(n_neighbors=5),
        "GradientBoosting": lambda: GradientBoostingClassifier(random_state=random_state),
        "NaiveBayes": lambda: GaussianNB(),
    }
    factory = _map.get(model_name)
    if factory is not None:
        return factory()

    # Optional: XGBoost / LightGBM
    if model_name == "XGBoost":
        from xgboost import XGBClassifier  # noqa: PLC0415
        return XGBClassifier(random_state=random_state, eval_metric="mlogloss", verbosity=0)
    if model_name == "LightGBM":
        from lightgbm import LGBMClassifier  # noqa: PLC0415
        return LGBMClassifier(random_state=random_state, verbosity=-1)

    return None


def _benchmark_632plus(
    X: np.ndarray,
    y: np.ndarray,
    use_pca: bool,
    n_pca_comps: int,
    verbose: bool,
    B: int = 200,
    bootstrap_ci: bool = False,
    no_info_method: str = "analytical",
    random_state: int = RANDOM_SEED,
) -> Tuple[pd.DataFrame, plt.Figure]:
    """.632+ bootstrap benchmark across models and feature spaces.

    When *use_pca* is True the PCA step is wrapped inside a
    ``sklearn.pipeline.Pipeline`` so that it is fitted **only on the
    in-bag sample** within each bootstrap iteration, preventing data
    leakage through the OOB samples.
    """
    from sklearn.metrics import accuracy_score, f1_score  # noqa: PLC0415
    from sklearn.pipeline import Pipeline  # noqa: PLC0415
    from scintilla.statistical_tests.bootstrap import dot632plus_bootstrap  # noqa: PLC0415

    pca_prefix = use_pca  # whether to also benchmark in PCA space
    n_c = min(n_pca_comps, X.shape[1] - 1, X.shape[0] - 1) if pca_prefix else 0

    metric_fns = {
        "accuracy": accuracy_score,
        "f1": lambda yt, yp: f1_score(yt, yp, average="macro", zero_division=0),
    }

    records = []
    space_defs = ["Gene"]
    if pca_prefix:
        space_defs.append("PCA")

    for space_name in space_defs:
        for model_name in _MODEL_FNS:
            if verbose:
                print(f"  .632+ | {space_name} | {model_name}")
            row: dict = {
                "space": space_name, "model": model_name,
                "status": "ok", "failure_reason": None,
            }
            try:
                estimator = _get_model_instance(model_name, random_state=random_state)
                if estimator is None:
                    continue  # skip models without a constructable instance

                # Wrap in a Pipeline for PCA space so that PCA is fitted
                # inside each bootstrap iteration (no data leakage).
                if space_name == "PCA":
                    estimator = Pipeline([
                        ("pca", PCA(n_components=n_c, random_state=random_state)),
                        ("clf", estimator),
                    ])

                for m_name, m_fn in metric_fns.items():
                    res = dot632plus_bootstrap(
                        estimator, X, y, m_fn, B=B,
                        no_info_method=no_info_method,
                        seed=random_state,
                    )
                    row[m_name] = res["estimate"]
                    if bootstrap_ci:
                        row[f"{m_name}_ci_low"] = res["ci_low"]
                        row[f"{m_name}_ci_high"] = res["ci_high"]
            except MemoryError:
                raise
            except Exception as e:
                if verbose:
                    print(f"    Failed: {e}")
                warnings.warn(f"{model_name} failed: {e}", stacklevel=2)
                row["status"] = "failed"
                row["accuracy"] = np.nan
                row["failure_reason"] = str(e)
            records.append(row)

    results_df = pd.DataFrame(records)

    fig, ax = plt.subplots(figsize=(12, 5))
    if "accuracy" in results_df.columns and results_df["accuracy"].notna().any():
        pivot = results_df.pivot_table(index="model", columns="space", values="accuracy")
        pivot.plot(kind="bar", ax=ax)
        ax.set_title("Classification Benchmark (.632+ Bootstrap)")
        ax.set_ylabel("Accuracy")
        ax.set_xlabel("Model")
        plt.xticks(rotation=45)
        plt.tight_layout()

    return results_df, fig


# ------------------------------------------------------------------
# Pairwise classifier comparison via paired bootstrap
# ------------------------------------------------------------------

def compare_classifiers(
    y_true: np.ndarray,
    predictions: dict,
    metric_fn=None,
    B: int = 2000,
    seed: int = RANDOM_SEED,
) -> pd.DataFrame:
    """Pairwise paired-bootstrap comparison of classifiers.

    Parameters
    ----------
    y_true:
        Ground-truth labels (1-D array).
    predictions:
        Mapping ``{model_name: y_pred_array}`` — one entry per classifier.
    metric_fn:
        ``metric_fn(y_true, y_pred) -> float``.  Defaults to accuracy.
    B:
        Bootstrap replicates.
    seed:
        Random seed.

    Returns
    -------
    DataFrame with columns ``model_a``, ``model_b``, ``diff`` (A − B),
    ``ci_low``, ``ci_high``, ``p_value``.
    """
    from sklearn.metrics import accuracy_score  # noqa: PLC0415
    from scintilla.statistical_tests.bootstrap import paired_bootstrap_test  # noqa: PLC0415

    if metric_fn is None:
        metric_fn = accuracy_score

    y_true = np.asarray(y_true)
    names = sorted(predictions.keys())
    records = []
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            res = paired_bootstrap_test(
                y_true, np.asarray(predictions[a]),
                np.asarray(predictions[b]),
                metric_fn, B=B, seed=seed,
            )
            records.append({
                "model_a": a,
                "model_b": b,
                "diff": res["diff"],
                "ci_low": res["ci_low"],
                "ci_high": res["ci_high"],
                "p_value": res["p_value"],
            })
    return pd.DataFrame(records)
