"""BenchmarkReport class for collecting and exporting benchmark results."""

from __future__ import annotations

import os
from typing import Dict, List, Optional

import numpy as np
import pandas as pd


class BenchmarkReport:
    """Collects benchmark results across stages and methods.

    Methods
    -------
    add_result(stage, method, metrics)
        Add a result entry.
    get_leaderboard(stage) -> DataFrame
        Get leaderboard for a specific stage sorted by main metric.
    export(output_dir)
        Save all results as CSV files.
    summary_table() -> DataFrame
        Get all stages and methods in one DataFrame.
    """

    def __init__(self):
        self._records: List[Dict] = []

    def add_result(self, stage: str, method: str, metrics: Dict) -> None:
        """Add a benchmark result.

        Parameters
        ----------
        stage:
            Pipeline stage name (e.g., 'clustering', 'classification').
        method:
            Method name.
        metrics:
            Dictionary of metric name -> value.
        """
        record = {"stage": stage, "method": method}
        record.update(metrics)
        self._records.append(record)

    def get_leaderboard(self, stage: str) -> pd.DataFrame:
        """Get leaderboard for a stage.

        Parameters
        ----------
        stage:
            Stage name to filter by.

        Returns
        -------
        pd.DataFrame sorted by first metric column descending.
        """
        df = self.summary_table()
        df = df[df["stage"] == stage].copy()
        metric_cols = [c for c in df.columns if c not in ("stage", "method")]
        if metric_cols:
            df = df.sort_values(metric_cols[0], ascending=False)
        return df.reset_index(drop=True)

    def export(self, output_dir: str) -> None:
        """Save results as CSV files, one per stage.

        Parameters
        ----------
        output_dir:
            Directory to save CSV files.
        """
        os.makedirs(output_dir, exist_ok=True)
        df = self.summary_table()
        if df.empty:
            return
        df.to_csv(os.path.join(output_dir, "benchmark_all.csv"), index=False)
        for stage in df["stage"].unique():
            stage_df = df[df["stage"] == stage]
            stage_df.to_csv(
                os.path.join(output_dir, f"benchmark_{stage}.csv"), index=False
            )

    def summary_table(self) -> pd.DataFrame:
        """Get all results as a single DataFrame.

        Returns
        -------
        pd.DataFrame with all stages and methods.
        """
        if not self._records:
            return pd.DataFrame(columns=["stage", "method"])
        return pd.DataFrame(self._records)
