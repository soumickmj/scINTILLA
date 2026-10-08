"""CLI differential expression subcommand."""


def add_args(parser):
    parser.add_argument("input", help="Path to input .h5ad file")
    parser.add_argument("--group-col", required=True, help="Group column in obs")
    parser.add_argument("--group1", required=True, help="First group")
    parser.add_argument("--group2", required=True, help="Second group")
    parser.add_argument("--method", default="wilcoxon", choices=["wilcoxon", "ttest", "permutation"])
    parser.add_argument("--output", default=None, help="Output CSV")
    parser.add_argument("--verbose", action="store_true")


def run(args):
    from scintilla.io.loaders import auto_detect_format

    adata = auto_detect_format(args.input)

    if args.method == "wilcoxon":
        from scintilla.differential_expression.wilcoxon import wilcoxon_de
        result = wilcoxon_de(adata, args.group_col, args.group1, args.group2)
    elif args.method == "ttest":
        from scintilla.differential_expression.ttest import ttest_de
        result = ttest_de(adata, args.group_col, args.group1, args.group2)
    elif args.method == "permutation":
        from scintilla.differential_expression.permutation import permutation_de
        result = permutation_de(adata, args.group_col, args.group1, args.group2)

    print(result.head(20).to_string(index=False))
    if args.output:
        result.to_csv(args.output, index=False)
