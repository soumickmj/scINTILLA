"""CLI generate-config subcommand."""


def add_args(parser):
    parser.add_argument(
        "--output",
        default="scintilla_config.yaml",
        help="Path to write the config file (default: scintilla_config.yaml)",
    )


def run(args):
    from scintilla.analysis_config import generate_default_yaml

    generate_default_yaml(args.output)
    print(f"Default config written to {args.output}")
    print("Edit the file then pass it with: scintilla <command> ... --config " + args.output)
