"""Mutual information based feature selection."""

from __future__ import annotations

from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_classif

from scintilla.config import RANDOM_SEED


def mi_feature_selection(
    X: np.ndarray,
    y: np.ndarray,
    n_features: int = 100,
    random_state: int = RANDOM_SEED,
) -> Tuple[np.ndarray, pd.DataFrame]:
    """Select top features by mutual information with the target.

    Parameters
    ----------
    X:
        Feature matrix (n_samples x n_features).
    y:
        Target labels.
    n_features:
        Number of top features to select.

    Returns
    -------
    selected_indices : np.ndarray of selected feature indices (sorted by score desc)
    scores_df : pd.DataFrame  columns=[feature_idx, mi_score]
    """
    scores = mutual_info_classif(X, y, random_state=random_state)
    n_sel = min(n_features, X.shape[1])
    selected_indices = np.argsort(scores)[::-1][:n_sel]
    scores_df = pd.DataFrame({
        "feature_idx": np.arange(len(scores)),
        "mi_score": scores,
    }).sort_values("mi_score", ascending=False).reset_index(drop=True)
    return selected_indices, scores_df
