"""CLI commands sub-package."""


def load_adata(args):
    """Load ``args.input`` and reduce a MuData to ``args.modality`` (if the file is .h5mu)."""
    from scintilla.io.loaders import auto_detect_format, ensure_anndata

    data = auto_detect_format(args.input)
    return ensure_anndata(data, modality=getattr(args, "modality", None))
