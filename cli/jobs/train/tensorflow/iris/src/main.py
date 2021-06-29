# imports
import os
import mlflow
import argparse

import pandas as pd
import matplotlib.pyplot as plt
import tensorflow_decision_forests as tfdf

# define functions
def main(args):
    # enable auto logging
    mlflow.autolog()

    # read in data
    df = pd.read_csv(args.iris_csv)

    # convert to tensorflow dataset
    ds = tfdf.keras.pd_dataframe_to_tf_dataset(df, label="species")

    # train model
    model = tfdf.keras.RandomForestModel().fit(ds)


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
    # add space in logs
    print("*" * 60)
    print("\n\n")

    # parse args
    args = parse_args()

    # run main function
    main(args)

    # add space in logs
    print("*" * 60)
    print("\n\n")
