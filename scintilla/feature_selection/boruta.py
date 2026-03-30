"""Boruta feature selection with fallback."""

from __future__ import annotations

from typing import Tuple

import numpy as np


def boruta_selection(
    X: np.ndarray,
    y: np.ndarray,
    n_estimators: int = 50,
    max_iter: int = 20,
    random_state: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """Feature selection using Boruta algorithm.

    Uses the ``boruta`` package if available, otherwise falls back to a
    permutation-based shadow feature approach.

    Parameters
    ----------
    X:
        Feature matrix (n_samples x n_features).
    y:
        Target labels.
    n_estimators:
        Number of trees in the Random Forest.
    max_iter:
        Maximum iterations (Boruta only).
    random_state:
        Random seed.

    Returns
    -------
    selected_mask : np.ndarray  bool mask, True = selected
    feature_importances : np.ndarray  importance score per feature
    """
    try:
        from boruta import BorutaPy  # noqa: PLC0415
        from sklearn.ensemble import RandomForestClassifier  # noqa: PLC0415

        rf = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state, n_jobs=-1)
        boruta = BorutaPy(rf, n_estimators="auto", max_iter=max_iter, random_state=random_state, verbose=0)
        boruta.fit(X, y)
        selected_mask = boruta.support_
        feature_importances = boruta.ranking_.astype(float)
        feature_importances = 1.0 / (feature_importances + 1e-8)
        return selected_mask, feature_importances

    except ImportError:
        return _shadow_feature_fallback(X, y, n_estimators=n_estimators, random_state=random_state)


def _shadow_feature_fallback(
    X: np.ndarray,
    y: np.ndarray,
    n_estimators: int = 100,
    random_state: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """Permutation-based shadow feature selection."""
    from sklearn.ensemble import RandomForestClassifier  # noqa: PLC0415

    rng = np.random.default_rng(random_state)
    n_features = X.shape[1]

    # Create shadow features: permute each column independently so that
    # within-feature value distributions are preserved but correlations
    # with the target are destroyed.
    X_shadow = np.column_stack([rng.permutation(X[:, j]) for j in range(X.shape[1])])
    X_aug = np.hstack([X, X_shadow])

    rf = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state, n_jobs=-1)
    rf.fit(X_aug, y)

    imp = rf.feature_importances_
    real_imp = imp[:n_features]
    shadow_imp = imp[n_features:]
    shadow_max = shadow_imp.max()

    selected_mask = real_imp > shadow_max
    return selected_mask, real_imp
