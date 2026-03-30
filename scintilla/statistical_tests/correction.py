"""Multiple testing correction utilities."""

from __future__ import annotations

import warnings
from typing import Tuple

import numpy as np
from statsmodels.stats.multitest import multipletests


def correct_pvalues(
    p_values: np.ndarray,
    method: str = "fdr_bh",
) -> Tuple[np.ndarray, np.ndarray]:
    """Apply multiple testing correction to p-values.

    Parameters
    ----------
    p_values:
        Array of p-values.
    method:
        Correction method: 'fdr_bh' (Benjamini-Hochberg), 'holm', 'bonferroni',
        'fdr_by' (Benjamini-Yekutieli).

    Returns
    -------
    rejected : np.ndarray  bool array of rejected hypotheses
    p_adjusted : np.ndarray  adjusted p-values
    """
    p_values = np.asarray(p_values, dtype=float)
    valid_methods = {"fdr_bh", "holm", "bonferroni", "fdr_by"}

    if method not in valid_methods:
        raise ValueError(f"method must be one of {valid_methods}, got '{method}'")

    rejected, p_adjusted, _, _ = multipletests(p_values, method=method)

    # Warn if statsmodels qvalue not available (informational only)
    try:
        from statsmodels.stats.multitest import local_fdr  # noqa: PLC0415
    except ImportError:
        warnings.warn(
            "Storey q-value not available. Using statsmodels multipletests instead.",
            ImportWarning,
            stacklevel=2,
        )

    return rejected, p_adjusted
