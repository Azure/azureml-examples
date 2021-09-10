# imports
import os
import gzip
import mlflow
import argparse

import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler

from adlfs import AzureBlobFileSystem as abfs

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

    # process data
    X_train, X_test, y_train, y_test = process_data(
        args.account_name, args.container_name
    )

    # train model
    model = train_model(params, X_train, X_test, y_train, y_test)


def read_images(f, num_images, image_size=28):
    f.read(16)  # magic

    buf = f.read(image_size * image_size * num_images)
    images = np.frombuffer(buf, dtype=np.uint8).astype(np.float32)
    images = images.reshape(num_images, image_size, image_size, 1)

    return images


def read_labels(f, num_labels):
    f.read(8)  # magic

    buf = f.read(num_labels)
    labels = np.frombuffer(buf, dtype=np.uint8)

    return labels


def process_data(df, random_state):
    # define constants
    train_len = 60000
    test_len = 10000

    # initialize filesystem
    container_name = args.container_name
    storage_options = {"account_name": args.account_name}

    fs = abfs(**storage_options)

    # read in data
    files = fs.ls(f"{container_name}/mnist")

    for f in files:
        if "train-images" in f:
            X_train = read_images(gzip.open(fs.open(f)), train_len)
        elif "train-labels" in f:
            y_train = read_labels(gzip.open(fs.open(f)), train_len)
        elif "images" in f:
            X_test = read_images(gzip.open(fs.open(f)), test_len)
        elif "labels" in f:
            y_test = read_labels(gzip.open(fs.open(f)), test_len)

    # flatten 2D image arrays into 1D
    X_train = X_train.reshape(train_len, -1)
    X_test = X_test.reshape(test_len, -1)

    # initialize and fit scaler
    scaler = MinMaxScaler()
    scaler.fit(X_train)

    # scale train and test data
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    # return data
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
    parser.add_argument("--account_name", type=str, default="azuremlexamples")
    parser.add_argument("--container_name", type=str, default="datasets")

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
