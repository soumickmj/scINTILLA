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
    from scintilla.classification.label_quality import label_quality
    from scintilla.classification.label_quality_variants import review_ranks
    from scintilla.cli.commands import load_adata
    from scintilla.io.exporters import save_anndata, save_results_csv

    config = AnalysisConfig.from_yaml(args.config) if args.config else None
    adata = load_adata(args)
    scores = label_quality(
        adata,
        cell_type_col=args.cell_type_col,
        use_rep=args.use_rep,
        supervised_use_rep=args.supervised_use_rep,
        config=config,
        fast=args.fast,
        random_state=args.seed,
        n_jobs=args.n_jobs,
        rerun=args.rerun,
        verbose=True if args.verbose else None,
    )
    out = Path(args.output)
    save_results_csv(scores, out)
    ranks_path = out.with_name(out.stem + "_ranks.csv")
    save_results_csv(review_ranks(scores), ranks_path)
    print(f"Scores ({len(scores)} labels x {scores.shape[1]} variants): {out}")
    print(f"Review ranks (1 = most suspicious): {ranks_path}")
    if args.save_h5ad:
        save_anndata(adata, args.save_h5ad)
