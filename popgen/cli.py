import os
import sys
import argparse
import logging
from popgen.project import popgen_run


def main(argv=None):
    """Command-line interface for running PopGen.

    Exit status: 0 = all scenarios succeeded, 1 = a scenario failed or the
    configuration could not be loaded, 2 = bad command-line usage.
    """
    from popgen import __version__

    parser = argparse.ArgumentParser(description="Run PopGen with a specified configuration file.")
    parser.add_argument(
        "config_path",
        type=str,
        help="Path to the configuration YAML file (e.g., configuration_arizona.yaml)"
    )
    parser.add_argument("--strict", action="store_true",
                        help="stop a scenario before IPF if the input checks find errors")
    parser.add_argument("--version", action="version", version=f"popgen3 {__version__}")
    args = parser.parse_args(argv)

    if not os.path.exists(args.config_path):
        logging.error(f"Configuration file '{args.config_path}' not found.")
        sys.exit(1)

    try:
        popgen_run(args.config_path, strict=True if args.strict else None)
    except Exception as e:  # details and traceback are already in the log
        print(f"popgen: FAILED - {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
