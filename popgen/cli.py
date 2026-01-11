import os
import argparse
import logging
from popgen.project import popgen_run

def main():
    """Command-line interface for running PopGen."""
    parser = argparse.ArgumentParser(description="Run PopGen with a specified configuration file.")
    parser.add_argument(
        "config_path",
        type=str,
        help="Path to the configuration YAML file (e.g., configuration_arizona.yaml)"
    )
    args = parser.parse_args()

    if not os.path.exists(args.config_path):
        logging.error(f"Configuration file '{args.config_path}' not found.")
        exit(1)

    popgen_run(args.config_path)

if __name__ == "__main__":
    main()
