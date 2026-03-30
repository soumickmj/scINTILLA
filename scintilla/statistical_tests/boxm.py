"""Box's M test for homogeneity of covariance matrices."""

from __future__ import annotations

import warnings
from typing import Dict, Union

import numpy as np
import pandas as pd
from scipy import stats

from scintilla.io.loaders import ensure_anndata


def box_m_test(
    data: Union[pd.DataFrame, "anndata.AnnData"],
    group_col: str,
) -> Dict:
    """Full Box's M test implementation.

    M = (n - g) * ln(|S_pooled|) - sum_i((n_i - 1) * ln(|S_i|))
    df = p(p+1)(g-1)/2

    Returns
    -------
    dict with keys: M_statistic, chi2_approx, p_value, df, recommendation ('LDA'|'QDA'),
    and optionally 'warning' when the approximation is unreliable.
    """
    import anndata as ad  # noqa: PLC0415

    adata = ensure_anndata(data)
    if group_col not in adata.obs.columns:
        raise KeyError(f"Column '{group_col}' not found in obs.")

    X = adata.X if not hasattr(adata.X, "toarray") else adata.X.toarray()
    X = X.astype(np.float64)
    groups = adata.obs[group_col].values
    unique_groups = np.unique(groups)

    p = X.shape[1]

    # Validate: groups with n_g < 2 cannot contribute a covariance estimate and
    # would cause a divide-by-zero in the c1 correction term.  Exclude them with
    # a warning; raise if fewer than 2 valid groups remain.
    group_sizes = {grp: int((groups == grp).sum()) for grp in unique_groups}
    small_groups = [grp for grp, sz in group_sizes.items() if sz < 2]
    if small_groups:
        warnings.warn(
            f"Box's M test: excluding {len(small_groups)} group(s) with fewer than "
            f"2 samples ({small_groups}). Adjust/merge groups to include them.",
            UserWarning,
            stacklevel=2,
        )
        unique_groups = np.array([grp for grp in unique_groups if grp not in small_groups])

    if len(unique_groups) < 2:
        raise ValueError(
            "Box's M test requires at least 2 groups each with >= 2 samples. "
            f"Only {len(unique_groups)} valid group(s) remain after excluding "
            "singleton groups."
        )

    g = len(unique_groups)
    n = int(sum(group_sizes[grp] for grp in unique_groups))

    group_covs = []
    group_ns = []
    singular_groups = []
    for grp in unique_groups:
        mask = groups == grp
        n_g = int(mask.sum())
        X_g = X[mask, :]
        cov_g = np.cov(X_g.T, ddof=1)
        if cov_g.ndim == 0:
            cov_g = np.array([[float(cov_g)]])
        # Check for singularity
        sign, _ = np.linalg.slogdet(cov_g)
        if sign <= 0:
            singular_groups.append(grp)
        group_covs.append(cov_g)
        group_ns.append(n_g)

    # If any group covariance is singular, the test is unreliable
    if singular_groups:
        warnings.warn(
            f"Box's M test: {len(singular_groups)} group(s) have singular "
            f"covariance matrices (p={p} may exceed group size). "
            "The test is unreliable; defaulting to QDA recommendation.",
            UserWarning,
            stacklevel=2,
        )
        return {
            "M_statistic": float("nan"),
            "chi2_approx": float("nan"),
            "p_value": float("nan"),
            "df": p * (p + 1) * (g - 1) // 2,
            "recommendation": "QDA",
            "warning": "singular_covariance",
        }

    # Pooled covariance
    S_pool = np.zeros((p, p))
    for i in range(g):
        S_pool += (group_ns[i] - 1) * group_covs[i]
    S_pool /= (n - g)

    def _logdet(M):
        sign, ld = np.linalg.slogdet(M)
        if sign <= 0:
            return float("nan")
        return ld

    log_det_pool = _logdet(S_pool)
    if np.isnan(log_det_pool):
        warnings.warn(
            "Box's M test: pooled covariance matrix is singular. "
            "The test is unreliable; defaulting to QDA recommendation.",
            UserWarning,
            stacklevel=2,
        )
        return {
            "M_statistic": float("nan"),
            "chi2_approx": float("nan"),
            "p_value": float("nan"),
            "df": p * (p + 1) * (g - 1) // 2,
            "recommendation": "QDA",
            "warning": "singular_pooled_covariance",
        }

    M_stat = (n - g) * log_det_pool - sum(
        (group_ns[i] - 1) * _logdet(group_covs[i]) for i in range(g)
    )

    df = p * (p + 1) * (g - 1) // 2
    c1 = (
        (sum(1.0 / (group_ns[i] - 1) for i in range(g)) - 1.0 / (n - g))
        * (2 * p ** 2 + 3 * p - 1) / (6 * (p + 1) * (g - 1))
    )

    # Guard: when c1 >= 1.0 (common in high-dimensional settings where
    # p >> n_g), the chi-squared approximation breaks down (Box, 1949).
    result: Dict = {}
    min_group_n = min(group_ns)
    if c1 >= 1.0 or p > min_group_n:
        warnings.warn(
            f"Box's M test: c1 correction factor = {c1:.4f} "
            f"(p={p}, min group size={min_group_n}). "
            "The chi-squared approximation is unreliable in this "
            "high-dimensional regime. Defaulting to QDA recommendation.",
            UserWarning,
            stacklevel=2,
        )
        result = {
            "M_statistic": float(M_stat),
            "chi2_approx": float("nan"),
            "p_value": float("nan"),
            "df": df,
            "recommendation": "QDA",
            "warning": "c1_approximation_breakdown",
        }
    else:
        chi2_approx = float(M_stat * (1 - c1))
        p_value = float(1 - stats.chi2.cdf(max(chi2_approx, 0), df=max(df, 1)))
        recommendation = "LDA" if p_value > 0.05 else "QDA"
        result = {
            "M_statistic": float(M_stat),
            "chi2_approx": chi2_approx,
            "p_value": p_value,
            "df": df,
            "recommendation": recommendation,
        }

    return result
