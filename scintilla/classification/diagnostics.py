"""Pre-classification diagnostics: normality and covariance homogeneity."""

from __future__ import annotations

import warnings
from typing import Dict, Union

import numpy as np
import pandas as pd
from scipy import stats

from scintilla.config import RANDOM_SEED
from scintilla.io.loaders import ensure_anndata


def check_normality_for_classifier(
    data: Union[pd.DataFrame, "anndata.AnnData"],
    target_col: str,
    alpha: float = 0.05,
    random_state: int = RANDOM_SEED,
) -> bool:
    """Per-group Shapiro-Wilk normality test.

    Returns True if the majority of (group, gene) pairs pass Shapiro-Wilk.
    Tests a random sample of up to 20 genes per group (seeded for
    reproducibility) rather than always testing the first 20.
    """
    import anndata as ad  # noqa: PLC0415

    adata = ensure_anndata(data, target_col=target_col)
    if target_col not in adata.obs.columns:
        raise KeyError(f"Column '{target_col}' not found in obs.")

    X = adata.X if not hasattr(adata.X, "toarray") else adata.X.toarray()
    X = X.astype(np.float64)
    groups = adata.obs[target_col].values
    unique_groups = np.unique(groups)

    rng = np.random.default_rng(random_state)
    passed = 0
    total = 0
    shapiro_error = None
    for grp in unique_groups:
        mask = groups == grp
        X_grp = X[mask, :]
        n_test = min(20, X_grp.shape[1])
        gene_indices = rng.choice(X_grp.shape[1], n_test, replace=False)
        for j in gene_indices:
            col = X_grp[:, j]
            if len(col) < 3:
                continue
            try:
                _, p = stats.shapiro(col[:5000])
                total += 1
                if p > alpha:
                    passed += 1
            except ValueError as exc:
                shapiro_error = shapiro_error or exc

    if shapiro_error is not None:
        warnings.warn(
            f"Classifier Shapiro-Wilk probe failed: {shapiro_error}",
            stacklevel=2,
        )

    return (passed / max(total, 1)) > 0.5


def covariance_homogeneity_test(
    data: Union[pd.DataFrame, "anndata.AnnData"],
    target_col: str,
) -> Dict:
    """Box's M test for homogeneity of covariance matrices.

    Returns
    -------
    dict with keys: M_statistic, chi2_approx, p_value, df, recommendation ('LDA' or 'QDA')
    """
    # Delegate to the canonical Box's M implementation
    from scintilla.statistical_tests.boxm import box_m_test  # noqa: PLC0415

    return box_m_test(data, group_col=target_col)


def influential_cells(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metric_fn=None,
    B: int = 2000,
    seed: int = RANDOM_SEED,
) -> Dict:
    """Identify influential cells via jackknife-after-bootstrap.

    Cells whose influence score exceeds 2× the inter-quartile range of all
    influence scores are flagged as influential — they may represent
    mislabelled or ambiguous cells.

    Parameters
    ----------
    y_true:
        Ground-truth labels for the test set.
    y_pred:
        Predicted labels for the test set.
    metric_fn:
        ``metric_fn(y_true, y_pred) -> float``.  Defaults to accuracy.
    B:
        Bootstrap replicates.
    seed:
        Random seed.

    Returns
    -------
    dict with:
        ``influence_scores`` — 1-D array of per-cell influence scores,
        ``influential_idx`` — indices of flagged cells,
        ``threshold`` — 2× IQR threshold used for flagging.
    """
    from sklearn.metrics import accuracy_score  # noqa: PLC0415
    from scintilla.statistical_tests.bootstrap import jackknife_after_bootstrap  # noqa: PLC0415

    if metric_fn is None:
        metric_fn = accuracy_score

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    paired = np.column_stack([y_true, y_pred])

    def _stat(arr: np.ndarray) -> float:
        return metric_fn(arr[:, 0], arr[:, 1])

    jab = jackknife_after_bootstrap(paired, _stat, B=B, seed=seed)
    scores = jab["influence_scores"]

    q1, q3 = np.percentile(scores, [25, 75])
    iqr = q3 - q1
    threshold = 2.0 * iqr
    influential_idx = np.where(np.abs(scores - np.median(scores)) > threshold)[0]

    return {
        "influence_scores": scores,
        "influential_idx": influential_idx,
        "threshold": float(threshold),
    }
