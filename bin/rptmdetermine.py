#! /usr/bin/env python3
"""
Validate PTM identifications derived from shotgun proteomics tandem mass
spectra.

"""
import argparse
import json

from rPTMDetermine import Retriever, RPTMDetermineConfig, Validator


def parse_args():
    """
    Parses the command line arguments to the script.

    Returns:
        The parsed command line arguments in an argparse Namespace.

    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "config",
        help=("The path to the JSON configuration file. "
              "See example_input.json for an example")
    )
    parser.add_argument(
        '--skip-validate',
        default=False,
        action='store_true',
        help=(
            'Skip validation and use the cached model. Note: This will fail '
            'if no cached model file exists for the current configuration.'
        )
    )
    return parser.parse_args()


def main():
    """
    The main entry point for the rPTMDetermine code.

    """
    args = parse_args()
    with open(args.config) as handle:
        conf = RPTMDetermineConfig(json.load(handle))

    if not args.skip_validate:
        validator = Validator(conf)
        validator.validate()

    retriever = Retriever(conf)
    retriever.retrieve()


if __name__ == '__main__':
    main()
