"""Script: run classification pipeline on an input file."""

import argparse
from scintilla.io.loaders import auto_detect_format
from scintilla.classification.run import supervised_analysis
from scintilla.io.exporters import save_results_json


def main():
    parser = argparse.ArgumentParser(description="Run classification pipeline")
    parser.add_argument("input", help="Input file path")
    parser.add_argument("--target-col", required=True, help="Target column in obs")
    parser.add_argument("--output", default="classification_summary.json")
    args = parser.parse_args()

    adata = auto_detect_format(args.input)
    result = supervised_analysis(adata, target_col=args.target_col)
    print(f"Best model: {result['best_model_name']}")
    summary = {k: v for k, v in result.items() if k not in ("best_model", "feature_importances")}
    save_results_json(summary, args.output)
    print(f"Results saved to {args.output}")


if __name__ == "__main__":
    main()
