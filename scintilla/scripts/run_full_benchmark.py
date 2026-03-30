"""Script to run the full benchmark suite."""

from __future__ import annotations

import argparse
import os


def main():
    parser = argparse.ArgumentParser(description="Run full scINTILLA benchmark")
    parser.add_argument("input", help="Path to input .h5ad file")
    parser.add_argument("--cell-type-col", default="cell_type")
    parser.add_argument("--output-dir", default="full_benchmark_output")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    from scintilla.io.loaders import auto_detect_format
    from scintilla.benchmarking.reporter import BenchmarkReport
    from scintilla.clustering.benchmark import benchmark_clustering_methods
    from scintilla.classification.benchmark import benchmark_models_comprehensive

    adata = auto_detect_format(args.input)
    report = BenchmarkReport()
    os.makedirs(args.output_dir, exist_ok=True)

    if args.cell_type_col in adata.obs.columns:
        print("Running clustering benchmark...")
        try:
            results_df, _, fig = benchmark_clustering_methods(
                adata, cell_type_col=args.cell_type_col, verbose=args.verbose
            )
            results_df.to_csv(os.path.join(args.output_dir, "clustering_results.csv"), index=False)
            fig.savefig(os.path.join(args.output_dir, "clustering_benchmark.png"), dpi=100)
            for _, row in results_df.iterrows():
                report.add_result("clustering", str(row.get("method", "?")), {"ari": float(row.get("ari", float("nan")))})
        except Exception as e:
            print(f"Clustering benchmark failed: {e}")

        print("Running classification benchmark...")
        try:
            results_df, fig = benchmark_models_comprehensive(
                adata, target_col=args.cell_type_col, verbose=args.verbose
            )
            results_df.to_csv(os.path.join(args.output_dir, "classification_results.csv"), index=False)
            fig.savefig(os.path.join(args.output_dir, "classification_benchmark.png"), dpi=100)
            for _, row in results_df.iterrows():
                report.add_result("classification", str(row.get("model", "?")), {"accuracy": float(row.get("accuracy", float("nan")))})
        except Exception as e:
            print(f"Classification benchmark failed: {e}")

    report.export(args.output_dir)
    print(f"\nFull benchmark complete. Results saved to {args.output_dir}")


if __name__ == "__main__":
    main()
