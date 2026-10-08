"""CLI dimensionality reduction subcommand."""


def add_args(parser):
    parser.add_argument("input", help="Path to input .h5ad file")
    parser.add_argument("--method", default="umap", choices=["umap", "tsne", "diffmap"])
    parser.add_argument("--use-rep", default="X_pca")
    parser.add_argument("--output", default=None, help="Output .h5ad file")
    parser.add_argument("--verbose", action="store_true")


def run(args):
    from scintilla.io.loaders import auto_detect_format
    from scintilla.io.exporters import save_anndata

    adata = auto_detect_format(args.input)

    if args.method == "umap":
        from scintilla.dimensionality_reduction.umap import run_umap
        adata = run_umap(adata, use_rep=args.use_rep)
    elif args.method == "tsne":
        from scintilla.dimensionality_reduction.tsne import run_tsne
        adata = run_tsne(adata, use_rep=args.use_rep)
    elif args.method == "diffmap":
        from scintilla.dimensionality_reduction.diffusion_map import run_diffusion_map
        adata = run_diffusion_map(adata, use_rep=args.use_rep)

    if args.output:
        save_anndata(adata, args.output)
    else:
        print(f"Embedding keys in obsm: {list(adata.obsm.keys())}")
