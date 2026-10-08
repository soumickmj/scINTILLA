"""Standalone classifier wrappers returning (model, metrics_dict)."""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import LinearSVC

from scintilla.config import RANDOM_SEED
from scintilla.evaluation.classification_metrics import calculate_metrics


def _fit_and_eval(model, X_train, X_test, y_train, y_test) -> Tuple[object, Dict]:
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test) if hasattr(model, "predict_proba") else None
    model_classes = getattr(model, "classes_", None)
    metrics = calculate_metrics(y_test, y_pred, y_prob, classes=model_classes)
    return model, metrics


def lda_classification(X_train, X_test, y_train, y_test) -> Tuple[object, Dict]:
    """Linear Discriminant Analysis classifier."""
    return _fit_and_eval(LinearDiscriminantAnalysis(), X_train, X_test, y_train, y_test)


def qda_classification(X_train, X_test, y_train, y_test) -> Tuple[object, Dict]:
    """Quadratic Discriminant Analysis classifier."""
    return _fit_and_eval(
        QuadraticDiscriminantAnalysis(reg_param=0.01), X_train, X_test, y_train, y_test
    )


def svm_classification(
    X_train, X_test, y_train, y_test, random_state: int = RANDOM_SEED,
) -> Tuple[object, Dict]:
    """Linear Support Vector Machine classifier."""
    return _fit_and_eval(
        LinearSVC(random_state=random_state, max_iter=2000),
        X_train, X_test, y_train, y_test,
    )


def random_forest_classification(
    X_train, X_test, y_train, y_test, n_estimators: int = 100,
    random_state: int = RANDOM_SEED,
) -> Tuple[object, Dict]:
    """Random Forest classifier."""
    return _fit_and_eval(
        RandomForestClassifier(n_estimators=n_estimators, random_state=random_state),
        X_train, X_test, y_train, y_test,
    )


def logistic_regression_classification(
    X_train, X_test, y_train, y_test, random_state: int = RANDOM_SEED,
) -> Tuple[object, Dict]:
    """Logistic Regression classifier."""
    return _fit_and_eval(
        LogisticRegression(max_iter=500, random_state=random_state, solver="lbfgs"),
        X_train, X_test, y_train, y_test,
    )


def mlp_classification(
    X_train, X_test, y_train, y_test, random_state: int = RANDOM_SEED,
) -> Tuple[object, Dict]:
    """Multi-Layer Perceptron classifier."""
    return _fit_and_eval(
        MLPClassifier(max_iter=500, random_state=random_state, hidden_layer_sizes=(100, 50)),
        X_train, X_test, y_train, y_test,
    )


def knn_classification(
    X_train, X_test, y_train, y_test, n_neighbors: int = 5
) -> Tuple[object, Dict]:
    """k-Nearest Neighbours classifier."""
    return _fit_and_eval(
        KNeighborsClassifier(n_neighbors=n_neighbors),
        X_train, X_test, y_train, y_test,
    )


def xgboost_classification(
    X_train, X_test, y_train, y_test, random_state: int = RANDOM_SEED,
) -> Tuple[object, Dict]:
    """XGBoost classifier."""
    try:
        from xgboost import XGBClassifier  # noqa: PLC0415
    except ImportError as exc:
        raise ImportError(
            "xgboost is required. Install with: pip install xgboost"
        ) from exc
    from sklearn.preprocessing import LabelEncoder  # noqa: PLC0415
    le = LabelEncoder()
    y_tr_enc = le.fit_transform(y_train)
    y_te_enc = le.transform(y_test)
    model = XGBClassifier(random_state=random_state, eval_metric="mlogloss", verbosity=0)
    model.fit(X_train, y_tr_enc)
    y_pred_enc = model.predict(X_test)
    y_prob = model.predict_proba(X_test) if hasattr(model, "predict_proba") else None
    y_pred = le.inverse_transform(y_pred_enc)
    # XGBoost classes_ are encoded ints; map back to original labels for alignment
    xgb_classes = le.inverse_transform(model.classes_)
    metrics = calculate_metrics(y_test, y_pred, y_prob, classes=xgb_classes)
    return model, metrics


def lightgbm_classification(
    X_train, X_test, y_train, y_test, random_state: int = RANDOM_SEED,
) -> Tuple[object, Dict]:
    """LightGBM classifier."""
    try:
        from lightgbm import LGBMClassifier  # noqa: PLC0415
    except ImportError as exc:
        raise ImportError(
            "lightgbm is required. Install with: pip install lightgbm"
        ) from exc
    model = LGBMClassifier(random_state=random_state, verbosity=-1)
    return _fit_and_eval(model, X_train, X_test, y_train, y_test)


def gradient_boosting_classification(
    X_train, X_test, y_train, y_test, random_state: int = RANDOM_SEED,
) -> Tuple[object, Dict]:
    """Histogram-based Gradient Boosting classifier (sklearn).

    Uses ``HistGradientBoostingClassifier`` which is orders of magnitude
    faster than the original ``GradientBoostingClassifier`` and supports
    native multi-core training.
    """
    from sklearn.ensemble import HistGradientBoostingClassifier  # noqa: PLC0415
    return _fit_and_eval(
        HistGradientBoostingClassifier(random_state=random_state),
        X_train, X_test, y_train, y_test,
    )


def naive_bayes_classification(X_train, X_test, y_train, y_test) -> Tuple[object, Dict]:
    """Gaussian Naive Bayes classifier."""
    from sklearn.naive_bayes import GaussianNB  # noqa: PLC0415
    return _fit_and_eval(GaussianNB(), X_train, X_test, y_train, y_test)


def stacking_ensemble_classification(
    X_train, X_test, y_train, y_test, random_state: int = RANDOM_SEED,
) -> Tuple[object, Dict]:
    """Stacking ensemble classifier (LogReg + RF + SVM base, LogReg final)."""
    from sklearn.ensemble import StackingClassifier  # noqa: PLC0415
    estimators = [
        ("lr", LogisticRegression(max_iter=500, random_state=random_state, solver="lbfgs")),
        ("rf", RandomForestClassifier(n_estimators=30, random_state=random_state)),
        ("svm", LinearSVC(random_state=random_state, max_iter=2000)),
    ]
    final_estimator = LogisticRegression(max_iter=500, random_state=random_state, solver="lbfgs")
    model = StackingClassifier(estimators=estimators, final_estimator=final_estimator, cv=5)
    return _fit_and_eval(model, X_train, X_test, y_train, y_test)
