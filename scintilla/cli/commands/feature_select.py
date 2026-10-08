"""CLI feature-select subcommand."""


def add_args(parser):
    parser.add_argument("input", help="Path to input .h5ad file")
    parser.add_argument("--n-per-pc", type=int, default=9, help="Top genes per PC")
    parser.add_argument("--output", default=None, help="Output gene list CSV")
    parser.add_argument("--config", default=None, help="Path to a scintilla YAML config file")
    parser.add_argument("--fast", action="store_true", help="Use fast preset")


def run(args):
    from scintilla.io.loaders import auto_detect_format
    from scintilla.feature_selection.pca_loadings import extract_top_genes_per_pc
    from scintilla.io.exporters import save_results_csv
    import pandas as pd

    adata = auto_detect_format(args.input)
    genes = extract_top_genes_per_pc(adata, n_per_pc=args.n_per_pc)
    print(f"Selected {len(genes)} genes:")
    print(genes)
    if args.output:
        save_results_csv(pd.DataFrame({"gene": genes}), args.output, index=False)
