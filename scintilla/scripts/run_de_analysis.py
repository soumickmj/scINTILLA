"""Script to run differential expression analysis."""

from __future__ import annotations

import argparse


def main():
    parser = argparse.ArgumentParser(description="Run differential expression analysis")
    parser.add_argument("input", help="Path to input .h5ad file")
    parser.add_argument("--group-col", required=True)
    parser.add_argument("--group1", required=True)
    parser.add_argument("--group2", required=True)
    parser.add_argument("--method", default="wilcoxon", choices=["wilcoxon", "ttest", "permutation"])
    parser.add_argument("--output", default="de_results.csv")
    args = parser.parse_args()

    from scintilla.io.loaders import auto_detect_format

    adata = auto_detect_format(args.input)

    if args.method == "wilcoxon":
        from scintilla.differential_expression.wilcoxon import wilcoxon_de
        result = wilcoxon_de(adata, args.group_col, args.group1, args.group2)
    elif args.method == "ttest":
        from scintilla.differential_expression.ttest import ttest_de
        result = ttest_de(adata, args.group_col, args.group1, args.group2)
    else:
        from scintilla.differential_expression.permutation import permutation_de
        result = permutation_de(adata, args.group_col, args.group1, args.group2)

    result.to_csv(args.output, index=False)
    print(f"Results saved to {args.output}")
    print(result.head(20).to_string(index=False))


if __name__ == "__main__":
    main()
