# imports
import os
import mlflow
import argparse

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# define functions
def main(args):
    # enable auto logging
    mlflow.autolog()

    # setup parameters
    params = {
        "penalty": args.penalty,
        "tol": args.tol,
        "C": args.C,
        "fit_intercept": args.fit_intercept,
        "intercept_scaling": args.intercept_scaling,
        "random_state": args.random_state,
        "solver": args.solver,
        "max_iter": args.max_iter,
        "multi_class": args.multi_class,
        "verbose": args.verbose,
    }

    # read in data
    df = pd.read_csv(args.diabetes_csv)

    # process data
    X_train, X_test, y_train, y_test = process_data(df, args.random_state)

    # train model
    model = train_model(params, X_train, X_test, y_train, y_test)


def process_data(df, random_state):
    # split dataframe into X and y
    X = df.drop(["target"], axis=1)
    y = df["target"]

    # train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=random_state
    )

    # return splits and encoder
    return X_train, X_test, y_train, y_test


def train_model(params, X_train, X_test, y_train, y_test):
    # train model
    model = LogisticRegression(**params)
    model = model.fit(X_train, y_train)

    # return model
    return model


def parse_args():
    # setup arg parser
    parser = argparse.ArgumentParser()

    # add arguments
    parser.add_argument("--diabetes-csv", type=str)
    parser.add_argument("--penalty", type=str, default="l2")
    parser.add_argument("--tol", type=float, default=1e-4)
    parser.add_argument("--C", type=float, default=1.0)
    parser.add_argument("--fit_intercept", type=bool, default=True)
    parser.add_argument("--intercept_scaling", type=float, default=1)
    parser.add_argument("--random_state", type=int, default=42)
    parser.add_argument("--solver", type=str, default="lbfgs")
    parser.add_argument("--max_iter", type=int, default=100)
    parser.add_argument("--multi_class", type=str, default="auto")
    parser.add_argument("--verbose", type=int, default=0)

    # parse args
    args = parser.parse_args()

    # return args
    return args


# run script
if __name__ == "__main__":
    # add space in logs
    print("\n\n")
    print("*" * 60)

    # parse args
    args = parse_args()

    # run main function
    main(args)

    # add space in logs
    print("*" * 60)
    print("\n\n")
