"""CLI full benchmark subcommand."""


def add_args(parser):
    parser.add_argument("input", help="Path to input .h5ad file")
    parser.add_argument("--cell-type-col", default="cell_type", help="Cell type column")
    parser.add_argument("--output-dir", default="benchmark_output")
    parser.add_argument("--verbose", action="store_true")


def run(args):
    from scintilla.io.loaders import auto_detect_format
    from scintilla.benchmarking.reporter import BenchmarkReport
    from scintilla.clustering.benchmark import benchmark_clustering_methods
    from scintilla.classification.benchmark import benchmark_models_comprehensive

    adata = auto_detect_format(args.input)
    report = BenchmarkReport()

    if args.cell_type_col in adata.obs.columns:
        try:
            results_df, _, _ = benchmark_clustering_methods(
                adata, cell_type_col=args.cell_type_col, verbose=args.verbose
            )
            for _, row in results_df.iterrows():
                report.add_result("clustering", row.get("method", "unknown"), {"ari": row.get("ari", float("nan"))})
        except Exception as e:
            if args.verbose:
                print(f"Clustering benchmark failed: {e}")

        try:
            results_df, _ = benchmark_models_comprehensive(
                adata, target_col=args.cell_type_col, verbose=args.verbose
            )
            for _, row in results_df.iterrows():
                report.add_result("classification", row.get("model", "unknown"), {"accuracy": row.get("accuracy", float("nan"))})
        except Exception as e:
            if args.verbose:
                print(f"Classification benchmark failed: {e}")

    report.export(args.output_dir)
    print(f"Benchmark results saved to {args.output_dir}")
    print(report.summary_table().head(20).to_string(index=False))
