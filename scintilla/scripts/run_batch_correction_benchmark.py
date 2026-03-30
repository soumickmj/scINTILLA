"""Script to run batch correction benchmark."""

from __future__ import annotations

import argparse
import sys


def main():
    parser = argparse.ArgumentParser(description="Run batch correction benchmark")
    parser.add_argument("input", help="Path to input .h5ad file")
    parser.add_argument("--batch-key", required=True, help="Batch column in obs")
    parser.add_argument("--label-key", default=None)
    parser.add_argument("--methods", nargs="+", default=None)
    parser.add_argument("--output-dir", default="batch_correction_output")
    args = parser.parse_args()

    from scintilla.io.loaders import auto_detect_format
    from scintilla.batch_correction.benchmark import benchmark_batch_correction
    import os

    adata = auto_detect_format(args.input)
    result = benchmark_batch_correction(
        adata,
        batch_key=args.batch_key,
        label_key=args.label_key,
        methods=args.methods,
    )
    os.makedirs(args.output_dir, exist_ok=True)
    result["leaderboard"].to_csv(
        os.path.join(args.output_dir, "batch_correction_leaderboard.csv"), index=False
    )
    print(f"Best method: {result['best_method']}")
    print(result["leaderboard"].to_string(index=False))


if __name__ == "__main__":
    main()
