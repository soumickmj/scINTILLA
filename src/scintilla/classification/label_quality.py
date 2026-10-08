"""One-call label-quality scoring: run both arms and score every label."""

from __future__ import annotations

from typing import Optional

import anndata as ad
import pandas as pd

from scintilla._logging import logger, resolve_verbose, verbosity_aware
from scintilla.analysis_config import AnalysisConfig


@verbosity_aware
def label_quality(
    adata: ad.AnnData,
    cell_type_col: str = "cell_type",
    *,
    use_rep: str = "X_pca",
    supervised_use_rep: Optional[str] = None,
    config: Optional[AnalysisConfig] = None,
    fast: bool = False,
    random_state: Optional[int] = None,
    n_jobs: int = 1,
    rerun: bool = False,
    verbose: Optional[bool] = None,
) -> pd.DataFrame:
    """Score how learnable and internally consistent every cell-type label is.

    Runs the unsupervised arm (a panel of clustering algorithms, neighbourhood confusion
    and fragmentation) and the supervised arm (classifier agreement, entropy and
    confidence) unless their results are already in ``adata.obs``, then scores each
    label with every scINTILLA variant (see
    :func:`~scintilla.classification.label_quality_variants.compute_label_quality_variants`).
    Every score is oriented so that **higher means a better label**; the labels most
    worth reviewing are those with the lowest scores.

    The per-cell columns of both arms are written to ``adata.obs`` (``scintilla_top*``
    and ``pred_*``); pass ``adata.copy()`` to keep the input unchanged.

    Parameters
    ----------
    adata
        Annotated data matrix.
    cell_type_col
        Column in ``adata.obs`` with the labels to assess.
    use_rep
        ``obsm`` embedding for clustering and Silhouette.  If it is ``"X_pca"`` and absent,
        PCA is computed and stored.
    supervised_use_rep
        ``obsm`` key for the classifiers; ``None`` uses the expression matrix.
    config
        Optional :class:`~scintilla.analysis_config.AnalysisConfig`; overrides ``fast``.
    fast
        Use the ``AnalysisConfig.fast()`` preset (fewer methods and models) when no
        ``config`` is given.  SHAP is always disabled.
    random_state
        Seed; overrides ``config.random_seed``.
    n_jobs
        Parallel workers for the clustering and classification benchmarks.
    rerun
        Recompute both arms even if their columns are already in ``adata.obs``.
    verbose
        Log progress at INFO level for this call.

    Returns
    -------
    pandas.DataFrame
        One row per label, one column per variant, higher = better.
    """
    from scintilla.classification.label_quality_variants import compute_label_quality_variants
    from scintilla.classification.run import supervised_analysis
    from scintilla.clustering.run import unsupervised_analysis

    cfg = config if config is not None else (AnalysisConfig.fast() if fast else AnalysisConfig.default())
    overrides = {"include_shap": False}
    if random_state is not None:
        overrides["random_seed"] = random_state
    cfg = cfg.copy(**overrides)
    verbose = resolve_verbose(verbose, cfg)

    obs = adata.obs
    conf_ranks = {c[: -len("_confusion")] for c in obs
                  if c.startswith("scintilla_top") and c.endswith("_confusion")}
    frag_ranks = {c[: -len("_fragmentation")] for c in obs
                  if c.startswith("scintilla_top") and c.endswith("_fragmentation")}
    has_unsup = "scintilla_top1" in conf_ranks and conf_ranks == frag_ranks
    has_sup = "pred_agreement" in obs and "pred_entropy" in obs

    pca_needed = use_rep not in adata.obsm
    if pca_needed and use_rep != "X_pca":
        raise KeyError(f"adata.obsm has no '{use_rep}'")
    if rerun or not has_unsup or pca_needed:
        logger.info("Running the unsupervised arm...")
        unsupervised_analysis(
            adata, cell_type_col=cell_type_col, use_rep=use_rep, run_pca_first=pca_needed,
            config=cfg, n_jobs=n_jobs, verbose=verbose,
        )
    if rerun or not has_sup:
        logger.info("Running the supervised arm...")
        supervised_analysis(
            adata, target_col=cell_type_col, use_rep=supervised_use_rep, check_consistency=True,
            include_shap=False, config=cfg, n_jobs=n_jobs, verbose=verbose,
        )
    return compute_label_quality_variants(adata, cell_type_col=cell_type_col, use_rep=use_rep)
