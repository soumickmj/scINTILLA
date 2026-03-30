"""CLI normalize (benchmark) subcommand."""


def add_args(parser):
    parser.add_argument("input", help="Path to input file")
    parser.add_argument("--output", default=None, help="Output CSV for benchmark results")
    parser.add_argument("--verbose", action="store_true")


def run(args):
    from scintilla.io.loaders import auto_detect_format
    from scintilla.preprocessing.benchmark import benchmark_transformations
    from scintilla.io.exporters import save_results_csv

    adata = auto_detect_format(args.input)
    results_df, best_name, _ = benchmark_transformations(adata, verbose=args.verbose)
    print(f"Best transformation: {best_name}")
    print(results_df[["transform", "composite_score"]].to_string(index=False))
    if args.output:
        save_results_csv(results_df, args.output)
