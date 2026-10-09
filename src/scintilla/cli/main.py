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
    from scintilla.cli.commands import (
        annotate,
        batch_correct,
        benchmark_all,
        classify,
        cluster,
        de,
        eda,
        estimate_time,
        feature_select,
        generate_config,
        label_quality,
        normalise,
        preprocess,
        reduce,
        run_all,
    )

    # Register subcommands
    eda.add_args(subparsers.add_parser("eda", help="Exploratory data analysis"))
    preprocess.add_args(subparsers.add_parser("preprocess", help="Preprocessing"))
    normalise.add_args(subparsers.add_parser("normalise", help="Normalisation benchmark"))
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
    label_quality.add_args(subparsers.add_parser("label-quality", help="Per-label scores for every scINTILLA variant"))

    for sub in subparsers.choices.values():
        if sub.prog.endswith("generate-config"):
            continue
        sub.add_argument("--modality", default=None,
                         help="Modality to analyse when the input is a .h5mu file (for example 'rna')")
        sub.add_argument("--quiet", action="store_true", help="Suppress progress messages")

    args = parser.parse_args(argv)
    if args.command is None:
        parser.print_help()
        sys.exit(0)

    # Dispatch
    dispatch = {
        "eda": eda.run,
        "preprocess": preprocess.run,
        "normalise": normalise.run,
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
        "label-quality": label_quality.run,
    }

    # The command line is not interactive: draw without a display, and show progress.
    import matplotlib

    matplotlib.use("Agg")
    from scintilla.settings import settings

    settings.verbosity = "warning" if getattr(args, "quiet", False) else "info"
    dispatch[args.command](args)


if __name__ == "__main__":
    main()
