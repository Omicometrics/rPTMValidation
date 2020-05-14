#! /usr/bin/env python3
"""
A module for retrieving missed identifications from ProteinPilot search
results.

"""
import argparse
import json

from rPTMDetermine import Retriever, RetrieverConfig


def parse_args():
    """
    Parses the command line arguments to the script.

    Returns:
        argparse.Namespace: The parsed command line arguments.

    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "config",
        help=("The path to the JSON configuration file. "
              "See example_input.json for an example"))
    return parser.parse_args()


def main():
    """
    The main entry point for the rPTMDetermine code.

    """
    args = parse_args()
    with open(args.config) as handle:
        conf = RetrieverConfig(json.load(handle))

    retriever = Retriever(conf)
    retriever.retrieve()


if __name__ == '__main__':
    main()
