"""CLI cluster subcommand."""


def add_args(parser):
    parser.add_argument("input", help="Path to input .h5ad file")
    parser.add_argument("--cell-type-col", default="cell_type", help="Cell type column in obs")
    parser.add_argument("--n-clusters", type=int, default=None)
    parser.add_argument("--output", default=None, help="Output CSV for results")
    parser.add_argument("--config", default=None, help="Path to a scintilla YAML config file")
    parser.add_argument("--fast", action="store_true", help="Use fast preset (fewer methods)")
    parser.add_argument("--verbose", action="store_true")


def run(args):
    from scintilla.io.loaders import auto_detect_format
    from scintilla.clustering.run import unsupervised_analysis
    from scintilla.io.exporters import save_results_csv
    from scintilla.analysis_config import AnalysisConfig

    cfg = None
    if args.config:
        cfg = AnalysisConfig.from_yaml(args.config)
    elif args.fast:
        cfg = AnalysisConfig.fast()

    adata = auto_detect_format(args.input)
    result = unsupervised_analysis(
        adata,
        cell_type_col=args.cell_type_col,
        n_clusters=args.n_clusters,
        verbose=args.verbose,
        config=cfg,
    )
    print(f"Best method: {result['best_method']}")
    print(result["results_df"].head(10).to_string(index=False))
    if args.output:
        save_results_csv(result["results_df"], args.output)
