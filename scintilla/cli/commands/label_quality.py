"""CLI label-quality subcommand: per-label scores for every scINTILLA variant."""


def add_args(parser):
    parser.add_argument("input", help="Path to input .h5ad file")
    parser.add_argument("--cell-type-col", default="cell_type", help="Label column in obs")
    parser.add_argument("--use-rep", default="X_pca",
                        help="obsm embedding for clustering and Silhouette (computed by PCA if X_pca is absent)")
    parser.add_argument("--supervised-use-rep", default=None,
                        help="obsm key for the classifiers (default: expression matrix)")
    parser.add_argument("--output", required=True, help="Output CSV of per-label scores")
    parser.add_argument("--save-h5ad", default=None, help="Also save the analysed AnnData here")
    parser.add_argument("--rerun", action="store_true",
                        help="Rerun both analyses even if their columns are already in obs")
    parser.add_argument("--config", default=None, help="Path to a scintilla YAML config file")
    parser.add_argument("--fast", action="store_true", help="Use fast preset (fewer models)")
    parser.add_argument("--seed", type=int, default=None, help="Random seed (overrides config)")
    parser.add_argument("--n-jobs", type=int, default=1)
    parser.add_argument("--verbose", action="store_true")


def run(args):
    from pathlib import Path

    from scintilla.analysis_config import AnalysisConfig
    from scintilla.classification.label_quality_variants import (
        compute_label_quality_variants, review_ranks)
    from scintilla.classification.run import supervised_analysis
    from scintilla.clustering.run import unsupervised_analysis
    from scintilla.io.exporters import save_anndata, save_results_csv
    from scintilla.io.loaders import auto_detect_format

    if args.config:
        cfg = AnalysisConfig.from_yaml(args.config)
    else:
        cfg = AnalysisConfig.fast() if args.fast else AnalysisConfig.default()
    overrides = {"include_shap": False}
    if args.seed is not None:
        overrides["random_seed"] = args.seed
    cfg = cfg.copy(**overrides)

    adata = auto_detect_format(args.input)
    obs = adata.obs
    conf_ranks = {c[:-len("_confusion")] for c in obs
                  if c.startswith("scintilla_top") and c.endswith("_confusion")}
    frag_ranks = {c[:-len("_fragmentation")] for c in obs
                  if c.startswith("scintilla_top") and c.endswith("_fragmentation")}
    has_unsup = "scintilla_top1" in conf_ranks and conf_ranks == frag_ranks
    has_sup = "pred_agreement" in obs and "pred_entropy" in obs

    pca_needed = args.use_rep not in adata.obsm
    if pca_needed and args.use_rep != "X_pca":
        raise KeyError(f"adata.obsm has no '{args.use_rep}'")
    if args.rerun or not has_unsup or pca_needed:
        adata = unsupervised_analysis(adata, cell_type_col=args.cell_type_col, use_rep=args.use_rep,
                                      run_pca_first=pca_needed, config=cfg, n_jobs=args.n_jobs,
                                      verbose=args.verbose)["adata"]
    if args.rerun or not has_sup:
        supervised_analysis(adata, target_col=args.cell_type_col, use_rep=args.supervised_use_rep,
                            check_consistency=True, include_shap=False, config=cfg,
                            n_jobs=args.n_jobs, verbose=args.verbose)

    scores = compute_label_quality_variants(adata, cell_type_col=args.cell_type_col,
                                            use_rep=args.use_rep)
    out = Path(args.output)
    save_results_csv(scores, out)
    ranks_path = out.with_name(out.stem + "_ranks.csv")
    save_results_csv(review_ranks(scores), ranks_path)
    print(f"Scores ({len(scores)} labels x {scores.shape[1]} variants): {out}")
    print(f"Review ranks (1 = most suspicious): {ranks_path}")
    if args.save_h5ad:
        save_anndata(adata, args.save_h5ad)
