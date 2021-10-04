# imports
import os
import mlflow
import argparse

from random import random

# define functions
def main(args):
    # print inputs
    print(f"inputA: {args.inputA}")
    print(f"inputB: {args.inputB}")
    print(f"inputC: {args.inputC}")

    # log inputs as parameters
    mlflow.log_param("inputA", args.inputA)
    mlflow.log_param("inputB", args.inputB)
    mlflow.log_param("inputC", args.inputC)

    # log a random metric
    mlflow.log_metric("random_metric", random())


def parse_args():
    # setup arg parser
    parser = argparse.ArgumentParser()

    # add arguments
    parser.add_argument("--inputA", type=float, default=0.5)
    parser.add_argument("--inputB", type=str, default="A")
    parser.add_argument("--inputC", type=float, default=1.0)

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
