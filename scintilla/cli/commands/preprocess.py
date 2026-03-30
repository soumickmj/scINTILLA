"""CLI preprocess subcommand."""


def add_args(parser):
    parser.add_argument("input", help="Path to input file")
    parser.add_argument("--transform", default="log_shift_size_factor", help="Transformation name")
    parser.add_argument("--output", default="preprocessed.h5ad", help="Output .h5ad path")


def run(args):
    from scintilla.io.loaders import auto_detect_format
    from scintilla.preprocessing.transformations import TRANSFORM_REGISTRY
    from scintilla.io.exporters import save_anndata

    adata = auto_detect_format(args.input)
    fn = TRANSFORM_REGISTRY.get(args.transform)
    if fn is None:
        print(f"Unknown transform '{args.transform}'. Available: {list(TRANSFORM_REGISTRY.keys())}")
        return
    adata_t = fn(adata)
    save_anndata(adata_t, args.output)
    print(f"Saved preprocessed data to {args.output}")
