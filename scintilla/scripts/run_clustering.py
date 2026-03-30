"""Script: run clustering pipeline on an input file."""

import argparse
from scintilla.io.loaders import auto_detect_format
from scintilla.clustering.run import unsupervised_analysis
from scintilla.io.exporters import save_results_csv


def main():
    parser = argparse.ArgumentParser(description="Run clustering pipeline")
    parser.add_argument("input", help="Input file path")
    parser.add_argument("--cell-type-col", required=True, help="Cell type column")
    parser.add_argument("--output", default="clustering_results.csv")
    parser.add_argument("--n-clusters", type=int, default=None)
    args = parser.parse_args()

    adata = auto_detect_format(args.input)
    result = unsupervised_analysis(adata, cell_type_col=args.cell_type_col, n_clusters=args.n_clusters)
    save_results_csv(result["results_df"], args.output)
    print(f"Best method: {result['best_method']}")
    print(f"Results saved to {args.output}")


if __name__ == "__main__":
    main()
