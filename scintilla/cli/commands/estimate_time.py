"""CLI estimate-time subcommand.

Runs a micro-calibration on a subsample of the input data and prints a
table of predicted wall-clock times for each enabled benchmarking method.
Optionally filters the table to methods that fit within a time budget.
"""


def add_args(parser):
    parser.add_argument("input", help="Path to input .h5ad file")
    parser.add_argument(
        "--cell-type-col", default="cell_type",
        help="Column in obs used to infer the number of clusters/classes (default: cell_type)",
    )
    parser.add_argument(
        "--stages",
        default="clustering,classification",
        help=(
            "Comma-separated pipeline stages to estimate. "
            "Choices: clustering, classification, batch_correction, feature_selection "
            "(default: clustering,classification)"
        ),
    )
    parser.add_argument(
        "--calibration-cells", type=int, default=500,
        help="Number of cells used for the micro-calibration subsample (default: 500)",
    )
    parser.add_argument(
        "--config", default=None,
        help="Path to a scintilla YAML config file",
    )
    parser.add_argument(
        "--fast", action="store_true",
        help="Use the fast preset (fewer methods)",
    )
    parser.add_argument(
        "--max-minutes", type=float, default=None,
        help="If given, print only methods whose cumulative time fits within this budget",
    )
    parser.add_argument(
        "--output", default=None,
        help="Optional path to save the estimates table as a CSV file",
    )
    parser.add_argument("--verbose", action="store_true")


def run(args):
    from scintilla.io.loaders import auto_detect_format
    from scintilla.analysis_config import AnalysisConfig
    from scintilla.benchmarking.time_estimator import (
        estimate_benchmark_time,
        print_time_budget,
    )

    cfg = None
    if args.config:
        cfg = AnalysisConfig.from_yaml(args.config)
    elif args.fast:
        cfg = AnalysisConfig.fast()

    stages = [s.strip() for s in args.stages.split(",") if s.strip()]

    adata = auto_detect_format(args.input)

    estimates = estimate_benchmark_time(
        adata,
        config=cfg,
        stages=stages,
        cell_type_col=args.cell_type_col,
        calibration_cells=args.calibration_cells,
        verbose=args.verbose,
    )

    if args.max_minutes is not None:
        result = print_time_budget(estimates, max_minutes=args.max_minutes)
    else:
        print(estimates[["stage", "method", "complexity", "grid_size",
                          "estimated_time", "status"]].to_string(index=False))
        result = estimates

    if args.output:
        result.to_csv(args.output, index=False)
        print(f"\nEstimates saved to {args.output}")
