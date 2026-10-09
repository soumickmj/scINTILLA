"""CLI full benchmark subcommand."""


def add_args(parser):
    parser.add_argument("input", help="Path to input .h5ad file")
    parser.add_argument("--cell-type-col", default="cell_type", help="Cell type column")
    parser.add_argument("--output-dir", default="benchmark_output")
    parser.add_argument("--verbose", action="store_true")


def run(args):
    import sys

    from scintilla.benchmarking.reporter import BenchmarkReport
    from scintilla.classification.benchmark import benchmark_models_comprehensive
    from scintilla.cli.commands import load_adata
    from scintilla.clustering.benchmark import benchmark_clustering_methods

    adata = load_adata(args)
    report = BenchmarkReport()

    if args.cell_type_col in adata.obs.columns:
        try:
            results_df, _ = benchmark_clustering_methods(
                adata, cell_type_col=args.cell_type_col, verbose=args.verbose
            )
            for _, row in results_df.iterrows():
                report.add_result("clustering", row.get("method", "unknown"), {"ari": row.get("ari", float("nan"))})
        except Exception as e:  # a failed stage must be visible, not only with --verbose
            print(f"Clustering benchmark failed: {e}", file=sys.stderr)

        try:
            results_df = benchmark_models_comprehensive(
                adata, target_col=args.cell_type_col, verbose=args.verbose
            )
            for _, row in results_df.iterrows():
                report.add_result("classification", row.get("model", "unknown"), {"accuracy": row.get("accuracy", float("nan"))})
        except Exception as e:  # a failed stage must be visible, not only with --verbose
            print(f"Classification benchmark failed: {e}", file=sys.stderr)

    report.export(args.output_dir)
    print(f"Benchmark results saved to {args.output_dir}")
    print(report.summary_table().head(20).to_string(index=False))
