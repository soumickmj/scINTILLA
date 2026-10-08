"""CLI classify subcommand."""


def add_args(parser):
    parser.add_argument("input", help="Path to input file")
    parser.add_argument("--target-col", default="cell_type", help="Target column in obs")
    parser.add_argument("--output", default=None, help="Output JSON path")
    parser.add_argument("--no-shap", action="store_true", help="Skip SHAP analysis")
    parser.add_argument("--config", default=None, help="Path to a scintilla YAML config file")
    parser.add_argument("--fast", action="store_true", help="Use fast preset (fewer models)")
    parser.add_argument("--verbose", action="store_true")


def run(args):
    from scintilla.io.loaders import auto_detect_format
    from scintilla.classification.run import supervised_analysis
    from scintilla.io.exporters import save_results_json
    from scintilla.analysis_config import AnalysisConfig

    cfg = None
    if args.config:
        cfg = AnalysisConfig.from_yaml(args.config)
    elif args.fast:
        cfg = AnalysisConfig.fast()

    adata = auto_detect_format(args.input)
    result = supervised_analysis(
        adata,
        target_col=args.target_col,
        include_shap=not args.no_shap,
        verbose=args.verbose,
        config=cfg,
    )
    print(f"Best model: {result['best_model_name']}")
    if args.output:
        summary = {k: v for k, v in result.items() if k not in ("best_model", "feature_importances")}
        save_results_json(summary, args.output)
