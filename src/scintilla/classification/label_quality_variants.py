"""Per-label scores for every scINTILLA label-quality variant.

One call returns, for each cell-type label, the original composite, the
fragmentation-aware composite, their ablations and raw components, the
fixed rank fusions with Silhouette and the frozen learned fusions -- the
scINTILLA scorers of the V4 benchmark, computed exactly as there.

Every score is oriented so that HIGHER = BETTER label; the labels most
worth reviewing are those with the LOWEST scores.

Prerequisites in ``adata.obs`` (written by the standard pipeline):

* ``scintilla_top{k}_confusion`` and ``scintilla_top{k}_fragmentation`` from
  :func:`scintilla.clustering.run.unsupervised_analysis`;
* ``pred_agreement``, ``pred_entropy`` (and optionally
  ``pred_avg_confidence``) from
  :func:`scintilla.classification.run.supervised_analysis` with
  ``check_consistency=True``.

Silhouette, needed by the fusions, is computed on ``adata.obsm[use_rep]``
unless supplied.
"""

from typing import Dict, Optional

import anndata as ad
import numpy as np
import pandas as pd

from scintilla.classification.visualise import compute_label_quality_score

# Frozen logistic models (damage predictors) fitted once on the development
# perturbations (V1, all five atlases, equal weight per error type x class)
# and never refitted. Features are within-dataset midrank percentiles:
# r_s = original scINTILLA composite, r_g = Silhouette, r_sg = r_s * r_g.
FROZEN_FUSION: Dict[str, dict] = {
    "fusion_log": {
        "intercept": 2.9277555346275688,
        "coef": {"r_s": -2.806706408259526, "r_g": -6.399737857821906, "r_sg": 2.9943287500623796},
    },
    "fusion_add": {
        "intercept": 2.469469952903643,
        "coef": {"r_s": -1.8297248915285826, "r_g": -4.958684167833358},
    },
}

# Column name -> one-line description, in output order.
VARIANTS: Dict[str, str] = {
    "scintilla_composite": "original composite (confusion x1 each, supervised x2 each)",
    "scintilla_composite_frag": "original composite + fragmentation columns (x1 each)",
    "scintilla_composite__silhouette": "mean percentile rank of original composite and Silhouette",
    "scintilla_composite_frag__silhouette": "mean percentile rank of fragmentation composite and Silhouette",
    "fusion_add": "frozen logistic fusion of original composite and Silhouette ranks",
    "fusion_log": "frozen logistic fusion with rank interaction term",
    "scintilla_unsup_only": "ablation: confusion arm only",
    "scintilla_sup_only": "ablation: supervised-consistency arm only",
    "scintilla_unsup_frag_only": "ablation: confusion + fragmentation, no supervised arm",
    "scintilla_confusion_top1": "component: negative confusion, best clustering",
    "scintilla_confusion_mean": "component: negative confusion, mean of top clusterings",
    "scintilla_fragmentation_mean": "component: negative fragmentation, mean of top clusterings",
    "scintilla_pred_agreement": "component: classifier agreement",
    "scintilla_pred_entropy": "component: negative classifier prediction entropy",
    "scintilla_pred_avg_confidence": "component: mean classifier confidence",
    "silhouette": "input: per-label mean Silhouette (fusion input, not a scINTILLA variant)",
}


def quality_rank(score: pd.Series) -> pd.Series:
    """Midrank percentile ``(rank - 0.5) / n`` of finite scores; NaN stays NaN."""
    s = score.astype(float).round(12)
    s = s.where(np.isfinite(s))
    n = s.notna().sum()
    return (s.rank(method="average") - 0.5) / n if n else s


def label_silhouette(adata: ad.AnnData, cell_type_col: str = "cell_type",
                     use_rep: str = "X_pca") -> pd.Series:
    """Per-label mean of per-cell silhouette on ``adata.obsm[use_rep]``."""
    from sklearn.metrics import silhouette_samples

    if use_rep not in adata.obsm:
        raise KeyError(f"adata.obsm has no '{use_rep}'; pass use_rep= or silhouette=.")
    labels = adata.obs[cell_type_col].astype(str).to_numpy()
    sil = silhouette_samples(np.asarray(adata.obsm[use_rep]), labels)
    return pd.Series(sil, index=labels).groupby(level=0).mean()


def compute_label_quality_variants(
    adata: ad.AnnData,
    cell_type_col: str = "cell_type",
    use_rep: str = "X_pca",
    silhouette: Optional[pd.Series] = None,
    fusion_models: Optional[Dict[str, dict]] = None,
) -> pd.DataFrame:
    """Score every label with every scINTILLA variant (see :data:`VARIANTS`).

    Parameters
    ----------
    adata : ad.AnnData
        Data after ``unsupervised_analysis`` and ``supervised_analysis``
        (``check_consistency=True``) on *cell_type_col*.
    cell_type_col : str
        Column in ``adata.obs`` with the labels to score.
    use_rep : str
        Embedding for Silhouette; use the one given to ``unsupervised_analysis``.
    silhouette : pd.Series, optional
        Precomputed per-label Silhouette (index = label); skips computing it.
    fusion_models : dict, optional
        Learned-fusion coefficients; defaults to :data:`FROZEN_FUSION`.

    Returns
    -------
    pd.DataFrame
        One row per label, one column per variant, higher = better.
    """
    obs = adata.obs
    conf_cols = sorted(c for c in obs if c.startswith("scintilla_top") and c.endswith("_confusion"))
    frag_cols = sorted(c for c in obs if c.startswith("scintilla_top") and c.endswith("_fragmentation"))
    missing = [name for name, ok in [("scintilla_top1_confusion", "scintilla_top1_confusion" in obs),
                                     ("scintilla_top*_confusion", conf_cols),
                                     ("scintilla_top*_fragmentation", frag_cols),
                                     ("pred_agreement", "pred_agreement" in obs),
                                     ("pred_entropy", "pred_entropy" in obs)] if not ok]
    if missing:
        raise ValueError(f"adata.obs lacks {missing}: run unsupervised_analysis and "
                         "supervised_analysis(check_consistency=True) on this branch first.")

    tmp_key = "_scintilla_variant_tmp"
    saved_tmp = obs[tmp_key].copy(deep=True) if tmp_key in obs else None

    def q(**weights):
        return compute_label_quality_score(adata, cell_type_col=cell_type_col, force=True,
                                           obs_key=tmp_key, **weights)

    try:
        out = pd.DataFrame({
            "scintilla_composite": q(fragmentation_weight=0.0),
            "scintilla_composite_frag": q(),
            "scintilla_unsup_only": q(supervised_weight=0.0, fragmentation_weight=0.0),
            "scintilla_sup_only": q(unsupervised_weight=0.0, fragmentation_weight=0.0),
            "scintilla_unsup_frag_only": q(supervised_weight=0.0),
        })
    finally:
        if saved_tmp is not None:
            adata.obs[tmp_key] = saved_tmp
        elif tmp_key in adata.obs:
            del adata.obs[tmp_key]

    g = obs.groupby(cell_type_col, observed=True)
    out["scintilla_confusion_top1"] = -g["scintilla_top1_confusion"].mean()
    out["scintilla_confusion_mean"] = -g[conf_cols].mean().mean(axis=1)
    out["scintilla_fragmentation_mean"] = -g[frag_cols].mean().mean(axis=1)
    out["scintilla_pred_agreement"] = g["pred_agreement"].mean()
    out["scintilla_pred_entropy"] = -g["pred_entropy"].mean()
    if "pred_avg_confidence" in obs:
        out["scintilla_pred_avg_confidence"] = g["pred_avg_confidence"].mean()

    if silhouette is None:
        silhouette = label_silhouette(adata, cell_type_col, use_rep)
    out.index = out.index.astype(str)
    silhouette = silhouette.copy()
    silhouette.index = silhouette.index.astype(str)
    out["silhouette"] = silhouette.reindex(out.index).astype(float)

    r_g = quality_rank(out["silhouette"])
    for scorer in ["scintilla_composite", "scintilla_composite_frag"]:
        out[f"{scorer}__silhouette"] = ((quality_rank(out[scorer]) + r_g) / 2).round(12)
    r_s = quality_rank(out["scintilla_composite"])
    features = {"r_s": r_s, "r_g": r_g, "r_sg": r_s * r_g}
    for name, model in (fusion_models or FROZEN_FUSION).items():
        out[name] = -(model["intercept"] + sum(c * features[k] for k, c in model["coef"].items()))

    out.index.name = cell_type_col
    return out[[c for c in VARIANTS if c in out]]


def review_ranks(scores: pd.DataFrame) -> pd.DataFrame:
    """Rank labels per variant: 1 = lowest score = most worth reviewing."""
    return scores.rank(method="min", ascending=True).astype("Int64")
