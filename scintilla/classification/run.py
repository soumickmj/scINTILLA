"""Supervised analysis dispatcher with decision logic."""

from __future__ import annotations

import warnings
from typing import Optional, Union

import anndata as ad
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tqdm.auto import tqdm

from scintilla.config import RANDOM_SEED, DEFAULT_TEST_SIZE, DEFAULT_CV_FOLDS
from scintilla.io.loaders import ensure_anndata
from scintilla.classification.models import (
    lda_classification,
    qda_classification,
    random_forest_classification,
    knn_classification,
    logistic_regression_classification,
    svm_classification,
    mlp_classification,
    gradient_boosting_classification,
    naive_bayes_classification,
    stacking_ensemble_classification,
)
from scintilla.classification.diagnostics import (
    check_normality_for_classifier,
    covariance_homogeneity_test,
)
from scintilla.classification.feature_importance import shap_analysis


def supervised_analysis(
    data: Union[pd.DataFrame, ad.AnnData],
    target_col: str = "cell_type",
    use_rep: Optional[str] = None,
    scale: bool = False,
    normality: Optional[bool] = None,
    test_size: Optional[float] = None,
    include_shap: Optional[bool] = None,
    check_consistency: bool = False,
    models: Optional[list] = None,
    n_jobs: int = 1,
    verbose: Optional[bool] = None,
    config=None,
    random_state: Optional[int] = None,
) -> dict:
    """Full supervised analysis pipeline with decision logic.

    Checks normality of the data; if the data are not normal an attempt is
    made to achieve normality via a Box-Cox transform.  All available
    classifiers (LogReg, RF, SVM, MLP, LDA, QDA, kNN) are then trained and
    evaluated; the one with the highest macro-F1 score is returned as the
    best model.  Optionally, SHAP feature importances are computed for the
    best model.

    Parameters
    ----------
    models:
        Optional list of model names to run.  Valid names:
        LogReg, RF, SVM, MLP, LDA, QDA, kNN, GradientBoosting,
        NaiveBayes, StackingEnsemble, XGBoost, LightGBM.
        When *None*, all available models are executed.
    config:
        Optional :class:`~scintilla.analysis_config.AnalysisConfig`.
        If provided, *config.classifiers* is used (unless *models*
        is explicitly given).

    Returns
    -------
    dict: best_model, best_model_name, all_results, feature_importances
    """
    from scintilla.preprocessing.normality import check_normality  # noqa: PLC0415
    from scintilla.preprocessing.transformations import box_cox_transform  # noqa: PLC0415

    if random_state is None:
        random_state = getattr(config, "random_seed", RANDOM_SEED) if config is not None else RANDOM_SEED
    if test_size is None:
        test_size = getattr(config, "test_size", DEFAULT_TEST_SIZE) if config is not None else DEFAULT_TEST_SIZE
    if verbose is None:
        verbose = getattr(config, "verbose", True) if config is not None else True

    adata = ensure_anndata(data, target_col=target_col)
    if target_col not in adata.obs.columns:
        raise KeyError(f"Column '{target_col}' not found in obs.")

    if use_rep is not None and use_rep in adata.obsm:
        X = adata.obsm[use_rep].astype(np.float64)
    else:
        X = adata.X if not hasattr(adata.X, "toarray") else adata.X.toarray()
        X = X.astype(np.float64)
    y = adata.obs[target_col].values

    # Determine normality
    if normality is None:
        if verbose:
            print("Checking normality...")
        normality, _ = check_normality(adata, random_state=random_state)

    if not normality:
        if verbose:
            print("Data not normal. Attempting Box-Cox transform...")
        try:
            adata_bc = box_cox_transform(adata)
            X_bc = adata_bc.X if not hasattr(adata_bc.X, "toarray") else adata_bc.X.toarray()
            X_bc = X_bc.astype(np.float64)
            adata_test = adata.copy()
            adata_test.X = X_bc
            normality, _ = check_normality(
                adata_test, random_state=random_state,
            )
            if normality:
                X = X_bc
                if verbose:
                    print("Box-Cox achieved normality.")
        except (RuntimeError, ValueError, np.linalg.LinAlgError, ArithmeticError):
            pass

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    if scale:
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X_tr)
        X_te = scaler.transform(X_te)

    all_model_fns = {
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

    # Optional models: offered only when their backend is installed, so a
    # missing xgboost/lightgbm is an omission rather than a failed row.
    from scintilla.classification.benchmark import _resolve_optional_models  # noqa: PLC0415
    all_model_fns.update(_resolve_optional_models())

    # Resolve which models to run: explicit param > config > all
    _selected_models = models
    if _selected_models is None and config is not None:
        _selected_models = getattr(config, "classifiers", None)
    if _selected_models is not None:
        all_model_fns = {k: v for k, v in all_model_fns.items() if k in _selected_models}

    # Resolve include_shap: explicit argument > config > historical default.
    if include_shap is None:
        include_shap = getattr(config, "include_shap", True) if config is not None else True

    def _run_one(name, fn, X_tr, X_te, y_tr, y_te):
        try:
            stochastic_models = {
                "LogReg", "RF", "SVM", "MLP", "XGBoost", "LightGBM",
                "GradientBoosting", "StackingEnsemble",
            }
            kwargs = {"random_state": random_state} if name in stochastic_models else {}
            model, metrics = fn(X_tr, X_te, y_tr, y_te, **kwargs)
            return name, {"model": model, "metrics": metrics}, None
        except MemoryError:
            raise
        except Exception as e:
            return name, None, str(e)

    all_results = {}
    if n_jobs == 1:
        pbar = tqdm(all_model_fns.items(), total=len(all_model_fns),
                    desc="Classification benchmark", disable=not verbose)
        for name, fn in pbar:
            pbar.set_postfix_str(name)
            name, result, err = _run_one(name, fn, X_tr, X_te, y_tr, y_te)
            if result is not None:
                all_results[name] = result
                if verbose:
                    print(f"  {name}: accuracy={result['metrics'].get('accuracy', 'N/A'):.3f}")
            else:
                all_results[name] = {
                    "status": "failed", "failure_reason": err,
                    "model": None, "metrics": {},
                }
                warnings.warn(f"{name} failed: {err}", stacklevel=2)
                if verbose:
                    print(f"  {name} failed: {err}")
        pbar.close()
    else:
        from joblib import Parallel, delayed  # noqa: PLC0415
        if verbose:
            print(f"Running {len(all_model_fns)} classifiers in parallel (n_jobs={n_jobs})...")
        outputs = Parallel(n_jobs=n_jobs, prefer="threads")(
            delayed(_run_one)(name, fn, X_tr, X_te, y_tr, y_te)
            for name, fn in all_model_fns.items()
        )
        for name, result, err in outputs:
            if result is not None:
                all_results[name] = result
                if verbose:
                    print(f"  {name}: accuracy={result['metrics'].get('accuracy', 'N/A'):.3f}")
            else:
                all_results[name] = {
                    "status": "failed", "failure_reason": err,
                    "model": None, "metrics": {},
                }
                warnings.warn(f"{name} failed: {err}", stacklevel=2)
                if verbose:
                    print(f"  {name} failed: {err}")

    # Pick best model by F1
    best_name = None
    best_f1 = -1.0
    for name, res in all_results.items():
        if res.get("status") == "failed":
            continue
        f1 = res["metrics"].get("f1", 0.0) or 0.0
        if f1 > best_f1:
            best_f1 = f1
            best_name = name

    best_model = all_results[best_name]["model"] if best_name else None

    # SHAP / feature importance for best model
    feature_importances = None
    if include_shap and best_model is not None:
        try:
            feat_names = list(adata.var_names) if adata.var_names is not None else None
            _, feature_importances = shap_analysis(
                best_model, X_tr, X_te, y_te, feature_names=feat_names,
                random_state=random_state,
            )
        except Exception as e:
            if verbose:
                print(f"  Feature importance failed: {e}")

    # ── Per-cell consistency check across all models ────────────────
    if check_consistency and all_results:
        if verbose:
            print("Computing per-cell prediction consistency...")
        X_all = X if not scale else StandardScaler().fit(X).transform(X)

        # Collect predictions & probabilities from each model on all cells
        from sklearn.preprocessing import LabelEncoder  # noqa: PLC0415
        le = LabelEncoder()
        le.fit(y)
        all_classes = le.classes_

        pred_matrix = []  # list of label arrays, one per model
        prob_dict = {}    # model_name -> (n_cells, n_classes) prob array

        for name, res in all_results.items():
            if res.get("status") == "failed":
                continue
            model = res["model"]
            try:
                # Some models (XGBoost/LightGBM) use encoded labels internally
                raw_pred = model.predict(X_all)
                # If predictions are integers and classes are strings, decode
                if hasattr(raw_pred, 'dtype') and np.issubdtype(raw_pred.dtype, np.integer) and not np.issubdtype(all_classes.dtype, np.integer):
                    raw_pred = le.inverse_transform(raw_pred)
                adata.obs[f"pred_{name}"] = pd.Categorical(
                    np.asarray(raw_pred).astype(str), categories=[str(c) for c in all_classes]
                )
                pred_matrix.append(np.asarray(raw_pred).astype(str))

                if hasattr(model, "predict_proba"):
                    probs = model.predict_proba(X_all)
                    prob_dict[name] = probs
                    # Store max probability as confidence
                    adata.obs[f"pred_{name}_confidence"] = probs.max(axis=1)
            except Exception as e:
                if verbose:
                    print(f"  Prediction for {name} failed: {e}")

        if pred_matrix:
            pred_arr = np.array(pred_matrix)  # (n_models, n_cells)
            n_models_ok = pred_arr.shape[0]
            n_cells = pred_arr.shape[1]

            # Per-cell: fraction of models agreeing with the mode prediction
            agreement = np.empty(n_cells, dtype=np.float64)
            mode_labels = np.empty(n_cells, dtype=object)
            for i in range(n_cells):
                labels_i = pred_arr[:, i]
                unique, counts = np.unique(labels_i, return_counts=True)
                best_idx = np.argmax(counts)
                mode_labels[i] = unique[best_idx]
                agreement[i] = counts[best_idx] / n_models_ok

            adata.obs["pred_consensus"] = pd.Categorical(
                mode_labels.astype(str), categories=[str(c) for c in all_classes]
            )
            adata.obs["pred_agreement"] = agreement

            # Entropy of label distribution across models (higher = less consistent)
            entropy = np.empty(n_cells, dtype=np.float64)
            for i in range(n_cells):
                labels_i = pred_arr[:, i]
                _, counts = np.unique(labels_i, return_counts=True)
                probs_i = counts / counts.sum()
                entropy[i] = -np.sum(probs_i * np.log2(probs_i + 1e-12))
            adata.obs["pred_entropy"] = entropy

            # Average max-probability across models that support predict_proba
            if prob_dict:
                avg_confidence = np.mean(
                    [p.max(axis=1) for p in prob_dict.values()], axis=0
                )
                adata.obs["pred_avg_confidence"] = avg_confidence

    return {
        "adata": adata,
        "best_model": best_model,
        "best_model_name": best_name,
        "all_results": all_results,
        "feature_importances": feature_importances,
        "normality": normality,
    }
