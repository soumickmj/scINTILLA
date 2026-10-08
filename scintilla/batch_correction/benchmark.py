"""Batch correction benchmark."""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from scintilla.batch_correction.metrics import batch_asw, bio_conservation_score
from scintilla.config import RANDOM_SEED


def benchmark_batch_correction(
    adata,
    batch_key: str,
    label_key: Optional[str] = None,
    methods: Optional[List[str]] = None,
    n_pcs: Optional[int] = None,
    scoring_method: Optional[str] = None,
    config=None,
    random_state: Optional[int] = None,
) -> Dict:
    """Benchmark batch correction methods.

    Parameters
    ----------
    adata:
        AnnData with batch labels.
    batch_key:
        Column in obs with batch labels.
    label_key:
        Column in obs with cell type labels (for bio conservation).
    methods:
        Methods to try. Defaults to available methods.
    n_pcs:
        Number of PCA components.
    scoring_method:
        ``"single_metric"`` (default) sorts by batch_asw only.
        ``"rank_aggregate"`` ranks methods by Borda count over
        batch_asw **and** bio-conservation (requires *label_key*).
        ``"pareto"`` identifies Pareto-optimal methods (batch_asw vs
        bio-conservation).
    config:
        Optional :class:`~scintilla.analysis_config.AnalysisConfig`.
    random_state:
        Random seed.  Explicit input overrides ``config.random_seed``.

    Returns
    -------
    dict with leaderboard (DataFrame), best_method (str), corrected_adatas (dict)
    """
    if n_pcs is None:
        n_pcs = getattr(config, "n_pca_comps", 30) if config is not None else 30
    if scoring_method is None:
        scoring_method = (
            getattr(config, "scoring_method", "single_metric")
            if config is not None else "single_metric"
        )
    scoring_method = {
        "weighted": "single_metric",
        "borda": "rank_aggregate",
    }.get(scoring_method, scoring_method)
    if scoring_method not in {"single_metric", "rank_aggregate", "pareto"}:
        raise ValueError(
            "Unknown batch scoring_method "
            f"{scoring_method!r}; expected 'single_metric', "
            "'rank_aggregate'/'borda', or 'pareto'"
        )
    if random_state is None:
        random_state = (
            getattr(config, "random_seed", RANDOM_SEED)
            if config is not None else RANDOM_SEED
        )
    if methods is None:
        methods = ["combat", "harmony", "bbknn", "scanorama"]
    if not methods:
        # Without this the leaderboard is a columnless DataFrame and scoring
        # fails later with an opaque KeyError('batch_asw').
        raise ValueError(
            "benchmark_batch_correction requires at least one batch correction "
            "method; got an empty list"
        )

    records = []
    corrected_adatas: Dict = {}

    for method in methods:
        try:
            adata_corr = _apply_method(
                adata, method, batch_key, n_pcs, random_state,
            )
            asw = batch_asw(adata_corr, batch_key, label_key)
            row: Dict = {"method": method, "batch_asw": asw, "status": "ok"}

            # Bio-conservation metric when label_key is provided
            if label_key is not None:
                row["bio_conservation"] = bio_conservation_score(
                    adata_corr, label_key,
                )

            records.append(row)
            corrected_adatas[method] = adata_corr
        except Exception as e:
            records.append({"method": method, "batch_asw": float("nan"), "status": str(e)})

    leaderboard = pd.DataFrame(records)

    # ---- Scoring / ranking ------------------------------------
    if scoring_method in {"rank_aggregate", "borda"} and label_key is not None and "bio_conservation" in leaderboard.columns:
        from scintilla.statistical_tests.rank_aggregation import borda_count  # noqa: PLC0415

        score_cols = ["batch_asw", "bio_conservation"]
        scores_df = leaderboard.set_index("method")[score_cols].dropna()
        if not scores_df.empty:
            borda_df = borda_count(scores_df)
            leaderboard = leaderboard.merge(
                borda_df[["borda_score", "borda_rank"]], left_on="method", right_index=True, how="left",
            )
            leaderboard = leaderboard.sort_values("borda_rank")
        else:
            leaderboard = leaderboard.sort_values("batch_asw", ascending=False)

    elif scoring_method == "pareto" and label_key is not None and "bio_conservation" in leaderboard.columns:
        leaderboard = _add_pareto_flag(leaderboard, ["batch_asw", "bio_conservation"])
        leaderboard = leaderboard.sort_values("batch_asw", ascending=False)
    else:
        leaderboard = leaderboard.sort_values("batch_asw", ascending=False)

    best_method = None
    valid = leaderboard.dropna(subset=["batch_asw"])
    if not valid.empty:
        best_method = str(valid.iloc[0]["method"])

    return {
        "leaderboard": leaderboard,
        "best_method": best_method,
        "corrected_adatas": corrected_adatas,
    }


def _apply_method(
    adata, method: str, batch_key: str, n_pcs: int, random_state: int,
):
    if method == "combat":
        from scintilla.batch_correction.combat import combat_correct  # noqa: PLC0415
        return combat_correct(adata, batch_key=batch_key)
    elif method == "harmony":
        from scintilla.batch_correction.harmony import harmony_correct  # noqa: PLC0415
        return harmony_correct(
            adata,
            batch_key=batch_key,
            n_components=n_pcs,
            random_state=random_state,
        )
    elif method == "bbknn":
        from scintilla.batch_correction.bbknn import bbknn_correct  # noqa: PLC0415
        return bbknn_correct(
            adata,
            batch_key=batch_key,
            n_pcs=n_pcs,
            random_state=random_state,
        )
    elif method == "scanorama":
        from scintilla.batch_correction.scanorama import scanorama_correct  # noqa: PLC0415
        return scanorama_correct(
            adata, batch_key=batch_key, random_state=random_state,
        )
    else:
        raise ValueError(f"Unknown method: {method}")


def _add_pareto_flag(
    df: pd.DataFrame,
    objective_cols: List[str],
) -> pd.DataFrame:
    """Add a ``pareto_optimal`` boolean column (all objectives maximised)."""
    df = df.copy()
    vals = df[objective_cols].values.astype(float)
    n = len(vals)
    is_pareto = np.ones(n, dtype=bool)
    for i in range(n):
        if not np.isfinite(vals[i]).all():
            is_pareto[i] = False
            continue
        for j in range(n):
            if i == j or not np.isfinite(vals[j]).all():
                continue
            # j dominates i if j >= i on all objectives and j > i on at least one
            if np.all(vals[j] >= vals[i]) and np.any(vals[j] > vals[i]):
                is_pareto[i] = False
                break
    df["pareto_optimal"] = is_pareto
    return df
