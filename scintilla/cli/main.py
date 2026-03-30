"""Main CLI entry point for scintilla."""

import argparse
import sys


def main(argv=None):
    """Entry point for the scintilla command-line interface."""
    parser = argparse.ArgumentParser(
        prog="scintilla",
        description="Single-Cell INTegrated Inference, Labelling, and Landscape Analysis pipeline",
    )
    subparsers = parser.add_subparsers(dest="command", help="Sub-command to run")

    # Import command modules
    from scintilla.cli.commands import eda  # noqa: PLC0415
    from scintilla.cli.commands import preprocess  # noqa: PLC0415
    from scintilla.cli.commands import normalize  # noqa: PLC0415
    from scintilla.cli.commands import cluster  # noqa: PLC0415
    from scintilla.cli.commands import classify  # noqa: PLC0415
    from scintilla.cli.commands import feature_select  # noqa: PLC0415
    from scintilla.cli.commands import run_all  # noqa: PLC0415
    from scintilla.cli.commands import batch_correct  # noqa: PLC0415
    from scintilla.cli.commands import reduce  # noqa: PLC0415
    from scintilla.cli.commands import de  # noqa: PLC0415
    from scintilla.cli.commands import annotate  # noqa: PLC0415
    from scintilla.cli.commands import benchmark_all  # noqa: PLC0415
    from scintilla.cli.commands import generate_config  # noqa: PLC0415
    from scintilla.cli.commands import estimate_time  # noqa: PLC0415

    # Register subcommands
    eda.add_args(subparsers.add_parser("eda", help="Exploratory data analysis"))
    preprocess.add_args(subparsers.add_parser("preprocess", help="Preprocessing"))
    normalize.add_args(subparsers.add_parser("normalize", help="Normalization benchmark"))
    cluster.add_args(subparsers.add_parser("cluster", help="Clustering"))
    classify.add_args(subparsers.add_parser("classify", help="Classification"))
    feature_select.add_args(subparsers.add_parser("feature-select", help="Feature selection"))
    run_all.add_args(subparsers.add_parser("run-all", help="Run full pipeline"))
    batch_correct.add_args(subparsers.add_parser("batch-correct", help="Batch correction"))
    reduce.add_args(subparsers.add_parser("reduce", help="Dimensionality reduction"))
    de.add_args(subparsers.add_parser("de", help="Differential expression"))
    annotate.add_args(subparsers.add_parser("annotate", help="Cell type annotation"))
    benchmark_all.add_args(subparsers.add_parser("benchmark-all", help="Full benchmark"))
    generate_config.add_args(subparsers.add_parser("generate-config", help="Generate default YAML config"))
    estimate_time.add_args(subparsers.add_parser("estimate-time", help="Estimate benchmark wall-clock time"))

    args = parser.parse_args(argv)
    if args.command is None:
        parser.print_help()
        sys.exit(0)

    # Dispatch
    dispatch = {
        "eda": eda.run,
        "preprocess": preprocess.run,
        "normalize": normalize.run,
        "cluster": cluster.run,
        "classify": classify.run,
        "feature-select": feature_select.run,
        "run-all": run_all.run,
        "batch-correct": batch_correct.run,
        "reduce": reduce.run,
        "de": de.run,
        "annotate": annotate.run,
        "benchmark-all": benchmark_all.run,
        "generate-config": generate_config.run,
        "estimate-time": estimate_time.run,
    }
    dispatch[args.command](args)


if __name__ == "__main__":
    main()
