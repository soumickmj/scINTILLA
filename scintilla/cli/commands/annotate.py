"""CLI cell type annotation subcommand."""


def add_args(parser):
    parser.add_argument("input", help="Path to input .h5ad file")
    parser.add_argument("--groupby", default="leiden", help="Column for marker finding")
    parser.add_argument("--method", default="wilcoxon", help="DE method")
    parser.add_argument("--n-genes", type=int, default=50)
    parser.add_argument("--output", default=None, help="Output .h5ad file")
    parser.add_argument("--verbose", action="store_true")


def run(args):
    from scintilla.io.loaders import auto_detect_format
    from scintilla.annotation.rank_genes import find_marker_genes

    adata = auto_detect_format(args.input)
    markers_df = find_marker_genes(adata, groupby=args.groupby, method=args.method, n_genes=args.n_genes)
    print(markers_df.head(20).to_string(index=False))
    if args.output:
        from scintilla.io.exporters import save_anndata
        save_anndata(adata, args.output)
