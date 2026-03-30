"""Scalability sweep for benchmarking."""

from __future__ import annotations

from typing import Callable, Dict, List

import numpy as np
import pandas as pd

from scintilla.benchmarking.profiler import profile_method


def scalability_sweep(
    method_fn: Callable,
    adata,
    fractions: List[float] = None,
    n_seeds: int = 1,
    bootstrap_ci: bool = False,
    n_bootstrap: int = 2000,
    **kwargs,
) -> pd.DataFrame:
    """Run method_fn on increasing fractions of data and profile performance.

    Parameters
    ----------
    method_fn:
        Function to profile; called as method_fn(adata_sub, **kwargs).
    adata:
        Full AnnData object.
    fractions:
        Fractions of cells to use.
    n_seeds:
        Number of random subsamples per fraction.  When > 1, each
        fraction is run with *n_seeds* independent subsamples so that
        variability in timing can be assessed.
    bootstrap_ci:
        When ``n_seeds > 1``, bootstrap the per-seed timings to
        produce ``elapsed_ci_low/high`` and ``memory_ci_low/high``
        columns.
    n_bootstrap:
        Number of bootstrap replicates for CI estimation.
    **kwargs:
        Additional arguments passed to method_fn.

    Returns
    -------
    pd.DataFrame  columns=[fraction, n_cells, elapsed_seconds, peak_memory_mb]
    (plus CI columns when bootstrap_ci=True and n_seeds > 1).
    """
    if fractions is None:
        fractions = [0.1, 0.25, 0.5, 0.75, 1.0]

    rng = np.random.default_rng(42)
    n_total = adata.n_obs
    profiled = profile_method(method_fn)

    if n_seeds <= 1:
        # Original single-seed path
        records = []
        for frac in fractions:
            n = max(2, int(np.ceil(n_total * frac)))
            idx = rng.choice(n_total, size=n, replace=False)
            adata_sub = adata[idx].copy()
            try:
                bench = profiled(adata_sub, **kwargs)
                records.append({
                    "fraction": frac,
                    "n_cells": n,
                    "elapsed_seconds": bench.elapsed_seconds,
                    "peak_memory_mb": bench.peak_memory_mb,
                })
            except TypeError as e:
                raise TypeError(
                    f"method_fn does not accept the provided keyword arguments "
                    f"({list(kwargs.keys())}). Check the function signature: {e}"
                ) from e
            except MemoryError:
                raise
            except (RuntimeError, ValueError, np.linalg.LinAlgError, ArithmeticError):
                records.append({
                    "fraction": frac,
                    "n_cells": n,
                    "elapsed_seconds": float("nan"),
                    "peak_memory_mb": float("nan"),
                })
        return pd.DataFrame(records)

    # Multi-seed path
    records = []
    for frac in fractions:
        n = max(2, int(np.ceil(n_total * frac)))
        elapsed_vals = []
        memory_vals = []
        for _ in range(n_seeds):
            idx = rng.choice(n_total, size=n, replace=False)
            adata_sub = adata[idx].copy()
            try:
                bench = profiled(adata_sub, **kwargs)
                elapsed_vals.append(bench.elapsed_seconds)
                memory_vals.append(bench.peak_memory_mb)
            except MemoryError:
                raise
            except Exception:
                elapsed_vals.append(float("nan"))
                memory_vals.append(float("nan"))

        elapsed_arr = np.array(elapsed_vals, dtype=np.float64)
        memory_arr = np.array(memory_vals, dtype=np.float64)
        row: Dict = {
            "fraction": frac,
            "n_cells": n,
            "elapsed_seconds": float(np.nanmean(elapsed_arr)),
            "peak_memory_mb": float(np.nanmean(memory_arr)),
        }

        if bootstrap_ci:
            from scintilla.statistical_tests.bootstrap import bootstrap_resample_metrics  # noqa: PLC0415
            valid_e = elapsed_arr[np.isfinite(elapsed_arr)]
            valid_m = memory_arr[np.isfinite(memory_arr)]
            if len(valid_e) >= 2:
                ci_e = bootstrap_resample_metrics(valid_e, B=n_bootstrap)
                row["elapsed_ci_low"] = ci_e["ci_low"]
                row["elapsed_ci_high"] = ci_e["ci_high"]
            if len(valid_m) >= 2:
                ci_m = bootstrap_resample_metrics(valid_m, B=n_bootstrap)
                row["memory_ci_low"] = ci_m["ci_low"]
                row["memory_ci_high"] = ci_m["ci_high"]

        records.append(row)

    return pd.DataFrame(records)
