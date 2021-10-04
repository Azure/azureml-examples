# imports
import os
import argparse

import pandas as pd

# define functions
def main(args):
    # read in data
    df = pd.read_csv(args.iris_csv)

    # print first 5 lines
    print(df.head())

    # ensure outputs directory exists
    os.makedirs("outputs", exist_ok=True)

    # save data to outputs
    df.to_csv("outputs/iris.csv", index=False)


def parse_args():
    # setup arg parser
    parser = argparse.ArgumentParser()

    # add arguments
    parser.add_argument("--iris-csv", type=str)

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
