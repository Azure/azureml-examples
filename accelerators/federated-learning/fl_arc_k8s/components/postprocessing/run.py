"""run.py for mock components"""
import argparse
from distutils.dir_util import copy_tree


def get_arg_parser(parser=None):
    """Parse the command line arguments for merge using argparse

    Args:
        parser (argparse.ArgumentParser or CompliantArgumentParser):
        an argument parser instance

    Returns:
        ArgumentParser: the argument parser instance

    Notes:
        if parser is None, creates a new parser instance
    """
    # add arguments that are specific to the component
    if parser is None:
        parser = argparse.ArgumentParser(description=__doc__)

    for arg in ["--input_data", "--results"]:
        parser.add_argument(arg)
    return parser


if __name__ == "__main__":
    # get the arguments
    parser = get_arg_parser()
    args = parser.parse_args()
    args = vars(args)

    copy_tree(args["input_data"], args["results"])
