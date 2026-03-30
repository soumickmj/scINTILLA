"""SHAP-based and permutation-based feature importance."""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance

from scintilla.config import RANDOM_SEED


def shap_analysis(
    model,
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_test: Optional[np.ndarray] = None,
    feature_names: Optional[List[str]] = None,
) -> Tuple[Optional[np.ndarray], pd.DataFrame]:
    """Compute feature importance using SHAP (or permutation importance fallback).

    Parameters
    ----------
    model:
        Fitted sklearn-compatible estimator.
    X_train:
        Training data used to build the SHAP explainer background.
    X_test:
        Test data on which importances are evaluated.
    y_test:
        True labels for *X_test*.  Required for the permutation-importance
        fallback (when the ``shap`` package is not installed); if omitted the
        fallback returns zero importances.
    feature_names:
        Optional list of feature names for the output DataFrame.

    Returns
    -------
    shap_values : np.ndarray or None
    feature_importance_df : pd.DataFrame  columns=[feature, importance]
    """
    n_features = X_train.shape[1]
    if feature_names is None:
        feature_names = [f"feature_{i}" for i in range(n_features)]

    try:
        import shap  # noqa: PLC0415

        explainer = shap.Explainer(model, X_train)
        shap_values = explainer(X_test)
        # Take mean absolute SHAP values across samples and classes
        sv = shap_values.values
        if sv.ndim == 3:
            importance = np.abs(sv).mean(axis=(0, 2))
        else:
            importance = np.abs(sv).mean(axis=0)
        shap_array = sv
    except ImportError:
        # Fallback: permutation importance (needs true labels)
        try:
            if y_test is None:
                raise ValueError("y_test is required for permutation importance fallback.")
            result = permutation_importance(model, X_test, y_test, n_repeats=5, random_state=RANDOM_SEED)
            importance = result.importances_mean
        except Exception:
            importance = np.zeros(n_features)
        shap_array = None

    feature_importance_df = pd.DataFrame({
        "feature": feature_names[:len(importance)],
        "importance": importance[:n_features],
    }).sort_values("importance", ascending=False).reset_index(drop=True)

    return shap_array, feature_importance_df
