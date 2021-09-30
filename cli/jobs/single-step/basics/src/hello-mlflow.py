# imports
import os
import mlflow
import argparse

from random import random

# define functions
def main(args):
    os.system(f"echo 'hello world' > {args.world_file}")
    mlflow.log_param("hello_factor", args.hello_factor)
    mlflow.log_metric("hello_measure", random())
    mlflow.log_artifact("helloworld.txt")

def parse_args():
    # setup arg parser
    parser = argparse.ArgumentParser()

    # add arguments
    parser.add_argument("--hello-factor", type=str, default="world")
    parser.add_argument("--world-file", type=str, default="helloworld.txt")

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
