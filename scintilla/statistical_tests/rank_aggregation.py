"""Weight-free rank aggregation for benchmark leaderboards.

Replaces arbitrary weighted-sum composites with Borda count aggregation
where each metric independently ranks the methods.
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np
import pandas as pd


def borda_count(
    scores_df: pd.DataFrame,
    higher_is_better: Optional[List[bool]] = None,
) -> pd.DataFrame:
    """Borda-count rank aggregation.

    Each column of *scores_df* is treated as an independent metric.  Methods
    (rows) are ranked per metric and the ranks are summed.  The method with
    the **lowest** total Borda score is ranked first.

    Parameters
    ----------
    scores_df:
        DataFrame where rows are methods/transforms and columns are metrics.
        Index should be method names.
    higher_is_better:
        Per-column flag.  ``True`` → higher raw score gets rank 1.
        Defaults to ``True`` for every column.

    Returns
    -------
    DataFrame with original columns plus ``borda_score`` and ``borda_rank``,
    sorted by ``borda_rank`` ascending.
    """
    df = scores_df.copy()
    n_metrics = df.shape[1]

    if higher_is_better is None:
        higher_is_better = [True] * n_metrics

    rank_cols: list[str] = []
    for col, ascending in zip(df.columns, higher_is_better):
        rc = f"_rank_{col}"
        # ascending=True → rank 1 = smallest; so invert when higher_is_better
        df[rc] = df[col].rank(ascending=not ascending, method="average")
        rank_cols.append(rc)

    df["borda_score"] = df[rank_cols].sum(axis=1)
    df["borda_rank"] = df["borda_score"].rank(method="min").astype(int)
    df = df.drop(columns=rank_cols).sort_values("borda_rank")
    return df


def rank_aggregate(
    scores_df: pd.DataFrame,
    method: str = "borda",
    higher_is_better: Optional[List[bool]] = None,
) -> pd.DataFrame:
    """Dispatcher for rank-aggregation methods.

    Parameters
    ----------
    scores_df:
        DataFrame where rows are methods and columns are metrics.
    method:
        Aggregation method.  Currently only ``"borda"`` is supported.
    higher_is_better:
        Per-column flag passed to the aggregation function.

    Returns
    -------
    Aggregated DataFrame with added rank columns.
    """
    if method == "borda":
        return borda_count(scores_df, higher_is_better=higher_is_better)
    raise ValueError(f"Unknown aggregation method: {method!r}. Supported: 'borda'.")
