"""Script: run normalization benchmark on an input file."""

import argparse
from scintilla.io.loaders import auto_detect_format
from scintilla.preprocessing.benchmark import benchmark_transformations
from scintilla.io.exporters import save_results_csv


def main():
    parser = argparse.ArgumentParser(description="Run normalization benchmark")
    parser.add_argument("input", help="Input file path")
    parser.add_argument("--output", default="normalization_benchmark.csv")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    adata = auto_detect_format(args.input)
    results_df, best_name, _ = benchmark_transformations(adata, verbose=args.verbose)
    print(f"Best transformation: {best_name}")
    print(results_df[["transform", "composite_score"]].to_string(index=False))
    save_results_csv(results_df, args.output)
    print(f"Results saved to {args.output}")


if __name__ == "__main__":
    main()
