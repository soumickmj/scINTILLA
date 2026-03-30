"""Data-adaptive parameter selection methods.

Provides principled, parameter-free alternatives to hard-coded defaults
for PCA component counts, DBSCAN eps, and Leiden/Louvain resolution grids.
"""

from __future__ import annotations

from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Optimal PCA component selection
# ---------------------------------------------------------------------------

def gavish_donoho_threshold(
    singular_values: np.ndarray,
    n: int,
    p: int,
) -> int:
    """Gavish-Donoho optimal hard threshold for singular values.

    Under an i.i.d. Gaussian noise model, the optimal hard threshold that
    minimises the Frobenius-norm estimation error is
    ``ω(β) · median(singular_values)`` where ``β = min(n,p)/max(n,p)``.

    Parameters
    ----------
    singular_values:
        Singular values from SVD (sorted descending).
    n:
        Number of observations (cells).
    p:
        Number of variables (genes).

    Returns
    -------
    int — optimal number of components to retain (≥ 1).

    References
    ----------
    Gavish & Donoho (2014). *The optimal hard threshold for singular values
    is 4/sqrt(3)*. IEEE Trans. Inf. Theory 60(8), 5040–5053.
    """
    singular_values = np.asarray(singular_values, dtype=np.float64)
    beta = min(n, p) / max(n, p)

    # Optimal coefficient ω(β) ≈ 0.56 β³ - 0.95 β² + 1.82 β + 1.43
    # (polynomial approximation from Gavish & Donoho)
    omega = 0.56 * beta**3 - 0.95 * beta**2 + 1.82 * beta + 1.43

    sigma = float(np.median(singular_values))
    threshold = omega * sigma

    n_components = int(np.sum(singular_values > threshold))
    return max(1, n_components)


def marchenko_pastur_cutoff(
    singular_values: np.ndarray,
    n: int,
    p: int,
    sigma_method: str = "median",
) -> int:
    """Number of significant components via Marchenko-Pastur distribution.

    Eigenvalues exceeding the upper edge of the MP bulk
    ``λ+ = σ² (1 + √(p/n))²`` are considered significant, where σ² is
    estimated from the bulk eigenvalues.

    Parameters
    ----------
    singular_values:
        Singular values (sorted descending).
    n:
        Number of observations.
    p:
        Number of variables.
    sigma_method:
        How to estimate the noise variance σ².

        - ``"median"`` (default) — divide the median eigenvalue by
          ``(1 - √γ)²``.  Fast but slightly overestimates σ² because
          the MP median is above the lower edge, leading to a
          conservative (fewer components) threshold.
        - ``"trimmed_mean"`` — iteratively estimate σ² from the
          trimmed mean of eigenvalues in the bulk (below the current
          λ+ estimate).  More accurate for datasets where the
          signal-to-noise ratio is moderate.

    Returns
    -------
    int — number of significant components (≥ 1).
    """
    import warnings as _warnings  # noqa: PLC0415

    singular_values = np.asarray(singular_values, dtype=np.float64)
    eigenvalues = singular_values ** 2

    gamma = p / n

    if gamma > 0.8:
        _warnings.warn(
            f"marchenko_pastur_cutoff: gamma (p/n) = {gamma:.2f} > 0.8.  "
            f"When p ≈ n the noise-variance estimate becomes unreliable "
            f"because (1 - sqrt(gamma))^2 → 0, which may retain too many "
            f"components.  Consider using 'gavish_donoho' instead, which "
            f"is more robust in this regime.",
            stacklevel=2,
        )

    if sigma_method == "trimmed_mean":
        # Iterative bulk estimation: start with a rough sigma² from the
        # median, compute λ+, then re-estimate sigma² from eigenvalues
        # below λ+ (the "bulk").  Converges in a few iterations.
        med_eig = float(np.median(eigenvalues))
        sigma2 = med_eig / ((1 - np.sqrt(gamma)) ** 2 + 1e-10) if gamma < 1 else med_eig
        for _ in range(5):
            lp = sigma2 * (1 + np.sqrt(gamma)) ** 2
            bulk = eigenvalues[eigenvalues <= lp]
            if len(bulk) < 2:
                break
            sigma2_new = float(np.mean(bulk)) / ((1 + np.sqrt(gamma)) ** 2 / 4 + (1 - np.sqrt(gamma)) ** 2 / 4 + 0.5)
            # Simpler: mean of bulk ≈ sigma² * (1 + gamma) for MP
            sigma2_new = float(np.mean(bulk)) / (1 + gamma) if gamma < 1 else float(np.mean(bulk))
            if abs(sigma2_new - sigma2) / (sigma2 + 1e-10) < 0.01:
                sigma2 = sigma2_new
                break
            sigma2 = sigma2_new
    else:
        # Default: median-based estimate
        med_eig = float(np.median(eigenvalues))
        sigma2 = med_eig / ((1 - np.sqrt(gamma)) ** 2 + 1e-10) if gamma < 1 else med_eig

    lambda_plus = sigma2 * (1 + np.sqrt(gamma)) ** 2

    n_components = int(np.sum(eigenvalues > lambda_plus))
    return max(1, n_components)


# ---------------------------------------------------------------------------
# Elbow detection (DBSCAN eps estimation)
# ---------------------------------------------------------------------------

def kneedle_elbow(
    x: np.ndarray,
    y: np.ndarray,
    direction: str = "increasing",
    S: float = 1.0,
) -> int:
    """Detect the elbow/knee point in a curve.

    Uses the ``kneed`` package if available; otherwise falls back to a
    maximum-curvature (second-derivative) method.

    Parameters
    ----------
    x, y:
        1-D arrays defining the curve.
    direction:
        ``"increasing"`` or ``"decreasing"``.
    S:
        Sensitivity parameter for ``kneed.KneeLocator`` (ignored by
        fallback).

    Returns
    -------
    int — index into *x* / *y* of the detected elbow point.
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)

    try:
        from kneed import KneeLocator  # noqa: PLC0415

        kn = KneeLocator(
            x, y,
            curve="convex" if direction == "increasing" else "concave",
            direction=direction,
            S=S,
        )
        if kn.knee is not None:
            return int(np.argmin(np.abs(x - kn.knee)))
    except ImportError:
        pass

    # Fallback: maximum curvature via second derivative
    if len(y) < 3:
        return 0
    dy = np.gradient(y, x)
    ddy = np.gradient(dy, x)
    curvature = np.abs(ddy) / (1 + dy**2) ** 1.5
    return int(np.argmax(curvature))


# ---------------------------------------------------------------------------
# Adaptive resolution search for Leiden / Louvain
# ---------------------------------------------------------------------------

def adaptive_resolution_search(
    run_fn: Callable[[float], np.ndarray],
    metric_fn: Callable[[np.ndarray], float],
    coarse_grid: Sequence[float],
    refine_steps: int = 5,
) -> Dict[str, object]:
    """Two-pass bisection search for optimal community-detection resolution.

    Parameters
    ----------
    run_fn:
        ``run_fn(resolution) -> labels`` — runs clustering at the given
        resolution and returns cluster labels.
    metric_fn:
        ``metric_fn(labels) -> float`` — evaluates label quality (higher is
        better).
    coarse_grid:
        Initial resolution values to evaluate.
    refine_steps:
        Number of bisection refinement iterations around the coarse-grid
        optimum.

    Returns
    -------
    dict with ``best_resolution``, ``best_score``, ``all_scores``
    (list of ``(resolution, score)`` tuples).
    """
    coarse_grid = sorted(coarse_grid)
    all_scores: List[Tuple[float, float]] = []

    # --- coarse pass ---
    best_res = coarse_grid[0]
    best_score = -np.inf
    for res in coarse_grid:
        labels = run_fn(res)
        score = metric_fn(labels)
        all_scores.append((res, score))
        if score > best_score:
            best_score = score
            best_res = res

    # --- refinement pass (bisection around best) ---
    idx = coarse_grid.index(best_res)
    lo = coarse_grid[max(0, idx - 1)]
    hi = coarse_grid[min(len(coarse_grid) - 1, idx + 1)]

    for _ in range(refine_steps):
        mid_lo = (lo + best_res) / 2
        mid_hi = (best_res + hi) / 2
        for res in [mid_lo, mid_hi]:
            labels = run_fn(res)
            score = metric_fn(labels)
            all_scores.append((res, score))
            if score > best_score:
                best_score = score
                best_res = res
        # Narrow the bracket
        if best_res <= (lo + hi) / 2:
            hi = (lo + hi) / 2
        else:
            lo = (lo + hi) / 2

    return {
        "best_resolution": best_res,
        "best_score": best_score,
        "all_scores": sorted(all_scores),
    }


# ---------------------------------------------------------------------------
# Normalised Variation of Information stability
# ---------------------------------------------------------------------------

def _variation_of_information(
    labels_a: np.ndarray,
    labels_b: np.ndarray,
    return_entropies: bool = False,
) -> "float | Tuple[float, float, float]":
    """Variation of Information between two clusterings.

    Parameters
    ----------
    labels_a, labels_b:
        Cluster label arrays of equal length.
    return_entropies:
        If ``True``, return ``(VI, H_a, H_b)`` instead of just ``VI``.
    """
    from sklearn.metrics import mutual_info_score  # noqa: PLC0415

    n = len(labels_a)
    if n == 0:
        return (0.0, 0.0, 0.0) if return_entropies else 0.0

    # Entropies via contingency.  counts from np.unique are always > 0,
    # so np.log cannot encounter zero — no epsilon needed.
    _, counts_a = np.unique(labels_a, return_counts=True)
    _, counts_b = np.unique(labels_b, return_counts=True)
    p_a = counts_a / n
    p_b = counts_b / n
    H_a = float(-np.sum(p_a * np.log(p_a)))
    H_b = float(-np.sum(p_b * np.log(p_b)))
    mi = mutual_info_score(labels_a, labels_b)
    vi = float(H_a + H_b - 2 * mi)

    if return_entropies:
        return vi, H_a, H_b
    return vi


def nvi_stability(
    labels_at_resolutions: List[Tuple[float, np.ndarray]],
    normalise: str = "log_n",
) -> Dict[str, object]:
    """Normalised Variation of Information between adjacent-resolution clusterings.

    Identifies plateau regions where the clustering is stable.

    Parameters
    ----------
    labels_at_resolutions:
        List of ``(resolution, labels)`` tuples, sorted by resolution.
    normalise:
        Normalisation strategy for the Variation of Information.

        - ``"log_n"`` (default) — divide VI by ``log(n)``.
          Makes values comparable across dataset sizes.
        - ``"max_entropy"`` — divide VI by ``max(H(A), H(B))``.
          Makes values comparable across different numbers of
          clusters, since raw VI tends to grow with cluster count.

    Returns
    -------
    dict with:
    - ``nvi_scores``: list of ``(resolution_pair_midpoint, nvi)``
    - ``stable_resolution``: resolution at the most stable plateau
    """
    if len(labels_at_resolutions) < 2:
        res = labels_at_resolutions[0][0] if labels_at_resolutions else 0.0
        return {"nvi_scores": [], "stable_resolution": res}

    pairs = sorted(labels_at_resolutions, key=lambda t: t[0])
    nvi_scores: List[Tuple[float, float]] = []

    for i in range(len(pairs) - 1):
        res_a, lab_a = pairs[i]
        res_b, lab_b = pairs[i + 1]
        vi, H_a, H_b = _variation_of_information(lab_a, lab_b, return_entropies=True)
        n = len(lab_a)

        if normalise == "max_entropy":
            denom = max(H_a, H_b)
            nvi = vi / (denom + 1e-10) if denom > 0 else 0.0
        else:  # "log_n" (default)
            nvi = vi / (np.log(n) + 1e-10) if n > 1 else 0.0

        midpoint = (res_a + res_b) / 2
        nvi_scores.append((midpoint, nvi))

    # Find the most stable region (minimum NVI)
    min_idx = int(np.argmin([s[1] for s in nvi_scores]))
    # The stable resolution is the midpoint of the most stable pair
    stable_res = pairs[min_idx][0]

    return {
        "nvi_scores": nvi_scores,
        "stable_resolution": stable_res,
    }
