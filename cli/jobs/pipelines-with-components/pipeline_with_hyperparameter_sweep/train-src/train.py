# imports
import os
import mlflow
import argparse
import pandas as pd
from pathlib import Path

from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

# define functions
def main(args):
    # enable auto logging
    mlflow.autolog()

    # make sure output folders exist (important for sweep trials)
    os.makedirs(args.model_output, exist_ok=True)
    os.makedirs(args.test_data, exist_ok=True)

    # setup parameters
    params = {
        "C": args.C,
        "kernel": args.kernel,
        "degree": args.degree,
        "gamma": args.gamma,
        "coef0": args.coef0,
        "shrinking": args.shrinking,
        "probability": args.probability,
        "tol": args.tol,
        "cache_size": args.cache_size,
        "class_weight": args.class_weight,
        "verbose": args.verbose,
        "max_iter": args.max_iter,
        "decision_function_shape": args.decision_function_shape,
        "break_ties": args.break_ties,
        "random_state": args.random_state,
    }

    # read in data (pipeline passes a uri_file CSV)
    df = pd.read_csv(args.data)

    # process data
    X_train, X_test, y_train, y_test = process_data(df, args.random_state)

    # train model
    model = train_model(params, X_train, y_train)

    # compute metric for sweep objective
    y_pred = model.predict(X_test)
    f1 = f1_score(y_test, y_pred, average="weighted")

    # IMPORTANT: this name must match objective.primary_metric exactly
    mlflow.log_metric("training_f1_score", float(f1))

    # save model to MLflow format into output folder
    mlflow.sklearn.save_model(model, path=args.model_output)

    # save test data for downstream predict step
    X_test.to_csv(Path(args.test_data) / "X_test.csv", index=False)
    y_test.to_csv(Path(args.test_data) / "y_test.csv", index=False)


def process_data(df, random_state):
    # split dataframe into X and y
    X = df.drop(["species"], axis=1)
    y = df["species"]

    # train/test split
    return train_test_split(
        X, y, test_size=0.2, random_state=random_state
    )


def train_model(params, X_train, y_train):
    model = SVC(**params)
    return model.fit(X_train, y_train)


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data", type=str)
    parser.add_argument("--C", type=float, default=1.0)
    parser.add_argument("--kernel", type=str, default="rbf")
    parser.add_argument("--degree", type=int, default=3)
    parser.add_argument("--gamma", type=str, default="scale")
    parser.add_argument("--coef0", type=float, default=0)
    parser.add_argument("--shrinking", type=bool, default=False)
    parser.add_argument("--probability", type=bool, default=False)
    parser.add_argument("--tol", type=float, default=1e-3)
    parser.add_argument("--cache_size", type=float, default=1024)
    parser.add_argument("--class_weight", type=dict, default=None)
    parser.add_argument("--verbose", type=bool, default=False)
    parser.add_argument("--max_iter", type=int, default=-1)
    parser.add_argument("--decision_function_shape", type=str, default="ovr")
    parser.add_argument("--break_ties", type=bool, default=False)
    parser.add_argument("--random_state", type=int, default=42)

    parser.add_argument("--model_output", type=str, help="Path of output model")
    parser.add_argument("--test_data", type=str, help="Path of output test data")

    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
