"""MRMR (Minimum Redundancy Maximum Relevance) feature selection."""

from __future__ import annotations

from typing import Tuple

import numpy as np

from scintilla.config import RANDOM_SEED


def mrmr_selection(
    X: np.ndarray,
    y: np.ndarray,
    n_features: int = 50,
) -> Tuple[np.ndarray, np.ndarray]:
    """Select features using MRMR criterion.

    Uses ``mrmr`` package if available, otherwise a greedy mutual-information
    based approximation.

    Parameters
    ----------
    X:
        Feature matrix (n_samples x n_features).
    y:
        Target labels.
    n_features:
        Number of features to select.

    Returns
    -------
    selected_indices : np.ndarray  indices of selected features (in selection order)
    scores : np.ndarray  relevance scores for selected features
    """
    n_sel = min(n_features, X.shape[1])
    try:
        import mrmr  # noqa: PLC0415
        import pandas as pd  # noqa: PLC0415

        X_df = pd.DataFrame(X)
        y_s = pd.Series(y)
        selected = mrmr.mrmr_classif(X=X_df, y=y_s, K=n_sel)
        selected_indices = np.array(selected, dtype=int)
        scores = np.arange(n_sel, 0, -1, dtype=float)
        return selected_indices, scores

    except ImportError:
        return _greedy_mi_mrmr(X, y, n_sel)


def _greedy_mi_mrmr(
    X: np.ndarray,
    y: np.ndarray,
    n_features: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Greedy MRMR approximation using mutual information."""
    from sklearn.feature_selection import mutual_info_classif  # noqa: PLC0415
    from sklearn.metrics import mutual_info_score  # noqa: PLC0415

    # Subsample cells for speed — MI estimates are stable at 5000 cells
    max_cells = 5000
    if X.shape[0] > max_cells:
        rng = np.random.default_rng(RANDOM_SEED)
        idx_sub = rng.choice(X.shape[0], max_cells, replace=False)
        X_sub = X[idx_sub]
        y_sub = y[idx_sub]
    else:
        X_sub = X
        y_sub = y

    n_total = X_sub.shape[1]
    relevance = mutual_info_classif(X_sub, y_sub, random_state=RANDOM_SEED)

    # Pre-bin all features once to avoid recomputing inside inner loop.
    # Use 8 quantile bins to better capture bimodal distributions common
    # in scRNA-seq data.
    X_binned = np.empty_like(X_sub, dtype=np.int32)
    for j in range(n_total):
        pcts = np.percentile(X_sub[:, j], [12.5, 25, 37.5, 50, 62.5, 75, 87.5])
        X_binned[:, j] = np.digitize(X_sub[:, j], pcts)

    selected = []
    remaining = list(range(n_total))
    scores = []

    for _ in range(n_features):
        if not remaining:
            break
        if not selected:
            idx = int(np.argmax([relevance[i] for i in remaining]))
            best = remaining[idx]
            scores.append(relevance[best])
        else:
            # MRMR score = relevance - mean redundancy
            mrmr_scores = []
            for i in remaining:
                red = np.mean([
                    mutual_info_score(X_binned[:, i], X_binned[:, j])
                    for j in selected
                ])
                mrmr_scores.append(relevance[i] - red)
            best_idx = int(np.argmax(mrmr_scores))
            best = remaining[best_idx]
            scores.append(mrmr_scores[best_idx])

        selected.append(best)
        remaining.remove(best)

    return np.array(selected, dtype=int), np.array(scores)
