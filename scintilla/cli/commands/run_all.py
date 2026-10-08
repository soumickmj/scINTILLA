"""CLI run-all subcommand (full pipeline)."""


def add_args(parser):
    parser.add_argument("input", help="Path to input file")
    parser.add_argument("--target-col", default="cell_type", help="Target/cell-type column in obs")
    parser.add_argument("--output-dir", default="results", help="Output directory")
    parser.add_argument("--config", default=None, help="Path to a scintilla YAML config file")
    parser.add_argument("--fast", action="store_true", help="Use fast preset (fewer methods)")
    parser.add_argument("--verbose", action="store_true")


def run(args):
    import os
    from scintilla.io.loaders import auto_detect_format
    from scintilla.clustering.run import unsupervised_analysis
    from scintilla.classification.run import supervised_analysis
    from scintilla.io.exporters import save_results_csv, save_results_json
    from scintilla.analysis_config import AnalysisConfig

    cfg = None
    if args.config:
        cfg = AnalysisConfig.from_yaml(args.config)
    elif args.fast:
        cfg = AnalysisConfig.fast()

    os.makedirs(args.output_dir, exist_ok=True)
    adata = auto_detect_format(args.input)

    print("=== Clustering ===")
    clust_result = unsupervised_analysis(
        adata, cell_type_col=args.target_col, verbose=args.verbose, config=cfg,
    )
    save_results_csv(clust_result["results_df"], os.path.join(args.output_dir, "clustering_results.csv"))
    print(f"Best clustering method: {clust_result['best_method']}")

    print("=== Classification ===")
    cls_result = supervised_analysis(
        adata, target_col=args.target_col, verbose=args.verbose, config=cfg,
    )
    print(f"Best classifier: {cls_result['best_model_name']}")
    summary = {
        "best_model": cls_result["best_model_name"],
        "normality": cls_result["normality"],
    }
    save_results_json(summary, os.path.join(args.output_dir, "classification_summary.json"))
    print("Done.")
