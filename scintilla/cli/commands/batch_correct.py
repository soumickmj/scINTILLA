"""CLI batch correction subcommand."""


def add_args(parser):
    parser.add_argument("input", help="Path to input .h5ad file")
    parser.add_argument("--batch-key", required=True, help="Batch column in obs")
    parser.add_argument("--label-key", default=None, help="Cell type column in obs")
    parser.add_argument("--methods", nargs="+", default=None)
    parser.add_argument("--output", default=None, help="Output directory")
    parser.add_argument("--verbose", action="store_true")


def run(args):
    from scintilla.io.loaders import auto_detect_format
    from scintilla.batch_correction.benchmark import benchmark_batch_correction

    adata = auto_detect_format(args.input)
    result = benchmark_batch_correction(
        adata,
        batch_key=args.batch_key,
        label_key=args.label_key,
        methods=args.methods,
    )
    print(f"Best method: {result['best_method']}")
    print(result["leaderboard"].to_string(index=False))
    if args.output:
        import os
        os.makedirs(args.output, exist_ok=True)
        result["leaderboard"].to_csv(
            os.path.join(args.output, "batch_correction_leaderboard.csv"), index=False
        )
