"""CLI eda subcommand."""


def add_args(parser):
    parser.add_argument("input", help="Path to input file (.h5ad, .csv)")
    parser.add_argument("--group-col", default=None, help="Column for grouping")
    parser.add_argument("--output", default=None, help="Output JSON path")


def run(args):
    from scintilla.io.loaders import auto_detect_format
    from scintilla.eda.summary import dataset_summary
    from scintilla.io.exporters import save_results_json
    import json

    adata = auto_detect_format(args.input)
    summary = dataset_summary(adata)
    print(json.dumps({k: v for k, v in summary.items() if not isinstance(v, list)}, indent=2))
    if args.output:
        save_results_json(summary, args.output)
