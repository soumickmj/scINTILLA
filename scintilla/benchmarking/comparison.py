"""Pairwise statistical comparison of benchmark methods."""

from __future__ import annotations

from typing import Callable, Optional

import numpy as np
import pandas as pd

from scintilla.config import RANDOM_SEED


def pairwise_method_comparison(
    results: pd.DataFrame,
    metric_col: str,
    method_col: str = "method",
    y_true: Optional[np.ndarray] = None,
    predictions: Optional[dict] = None,
    test: str = "permutation",
    n_permutations: int = 1000,
    seed: int = RANDOM_SEED,
) -> pd.DataFrame:
    """Pairwise significance tests between benchmark methods.

    Parameters
    ----------
    results:
        DataFrame with at least *method_col* and *metric_col* columns.
        Each row is one run / seed / fold of a method.
    metric_col:
        Column containing the numeric metric to compare.
    method_col:
        Column identifying the method.
    y_true:
        Ground-truth labels (required for ``"mcnemar"`` and
        ``"paired_bootstrap"`` tests).
    predictions:
        ``{method_name: y_pred_array}`` (required for ``"mcnemar"`` and
        ``"paired_bootstrap"``).
    test:
        ``"permutation"`` — permutation test on per-run metric values.
        ``"mcnemar"`` — McNemar's test on classification predictions.
        ``"paired_bootstrap"`` — paired bootstrap on predictions.
    n_permutations:
        Permutation / bootstrap replicates.
    seed:
        Random seed.

    Returns
    -------
    DataFrame with ``method_a``, ``method_b``, ``diff``, ``p_value``,
    ``significant`` (at α = 0.05).
    """
    methods = sorted(results[method_col].unique())
    records = []

    for i, a in enumerate(methods):
        for b in methods[i + 1:]:
            if test == "permutation":
                vals_a = results.loc[results[method_col] == a, metric_col].dropna().values
                vals_b = results.loc[results[method_col] == b, metric_col].dropna().values
                diff, p = _permutation_diff(vals_a, vals_b, n_permutations, seed)
            elif test == "mcnemar":
                if y_true is None or predictions is None:
                    raise ValueError("y_true and predictions required for mcnemar test")
                from scintilla.statistical_tests.permutation import mcnemar_test  # noqa: PLC0415
                res = mcnemar_test(y_true, predictions[a], predictions[b])
                diff = float(np.mean(np.asarray(predictions[a]) == y_true)
                             - np.mean(np.asarray(predictions[b]) == y_true))
                p = res["p_value"]
            elif test == "paired_bootstrap":
                if y_true is None or predictions is None:
                    raise ValueError("y_true and predictions required for paired_bootstrap")
                from sklearn.metrics import accuracy_score  # noqa: PLC0415
                from scintilla.statistical_tests.bootstrap import paired_bootstrap_test  # noqa: PLC0415
                res = paired_bootstrap_test(
                    y_true, np.asarray(predictions[a]),
                    np.asarray(predictions[b]),
                    accuracy_score, B=n_permutations, seed=seed,
                )
                diff = res["diff"]
                p = res["p_value"]
            else:
                raise ValueError(f"Unknown test: {test}")

            records.append({
                "method_a": a,
                "method_b": b,
                "diff": float(diff),
                "p_value": float(p),
                "significant": p < 0.05,
            })

    return pd.DataFrame(records)


def _permutation_diff(
    vals_a: np.ndarray,
    vals_b: np.ndarray,
    n_perm: int,
    seed: int,
) -> tuple:
    """Two-sample permutation test on metric vectors."""
    rng = np.random.default_rng(seed)
    obs_diff = float(np.mean(vals_a) - np.mean(vals_b))
    pooled = np.concatenate([vals_a, vals_b])
    n_a = len(vals_a)
    count = 0
    for _ in range(n_perm):
        rng.shuffle(pooled)
        perm_diff = np.mean(pooled[:n_a]) - np.mean(pooled[n_a:])
        if abs(perm_diff) >= abs(obs_diff):
            count += 1
    p = (count + 1) / (n_perm + 1)  # continuity-corrected
    return obs_diff, p
