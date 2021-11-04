# imports
import os
import mlflow
import argparse

from random import random

# define functions
def main(args):
    # print inputs
    print(f"A: {args.A}")
    print(f"B: {args.B}")
    print(f"C: {args.C}")

    # log inputs as parameters
    mlflow.log_param("A", args.A)
    mlflow.log_param("B", args.B)
    mlflow.log_param("C", args.C)

    # log a random metric
    mlflow.log_metric("random_metric", random())


def parse_args():
    # setup arg parser
    parser = argparse.ArgumentParser()

    # add arguments
    parser.add_argument("--A", type=float, default=0.5)
    parser.add_argument("--B", type=str, default="hello world")
    parser.add_argument("--C", type=float, default=1.0)

    # parse args
    args = parser.parse_args()

    # return args
    return args


# run script
if __name__ == "__main__":
    # parse args
    args = parse_args()

    # run main function
    main(args)
