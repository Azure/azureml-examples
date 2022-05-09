# imports
from dataclasses import dataclass
import os
import mlflow
import argparse

import pandas as pd
from pathlib import Path

from sklearn.svm import SVC
from sklearn.model_selection import train_test_split

from pandas import DataFrame

from mlcomponent import command_component
from mlcomponent import Asset

# input/output load/dumpers
def load_data(path) -> DataFrame:
    # read in data
    df = pd.read_csv(path)
    return df


def save_data(data, path):
    X_test, y_test = data
    X_test.to_csv(Path(path) / "X_test.csv", index=False)
    y_test.to_csv(Path(path) / "y_test.csv", index=False)


def save_model(model, path):
    # Output the model and test data
    mlflow.sklearn.save_model(model, path + "/model")


@dataclass
class Outputs:
    model_output: Asset(type='mlflow_model', dump=save_data)
    test_data: Asset(type='uri_folder', dump=save_model)


@command_component(name='train_model')
def main(data: Asset(type="uri_folder", load=load_data),
         C=1.0, kernel='rbf', degree=3, gamma='scale', coef0:float=0, 
         shrinking=False, probability=False, tol=1e-3, cache_size=1024, class_weight:bool=None, 
         verbose=False, max_iter=-1, decision_function_shape='ovr', break_ties=False, random_state=42) -> Outputs:
    # enable auto logging
    mlflow.autolog()
    # setup parameters
    params = {
        "C": C,
        "kernel": kernel,
        "degree": degree,
        "gamma": gamma,
        "coef0": coef0,
        "shrinking": shrinking,
        "probability": probability,
        "tol": tol,
        "cache_size": cache_size,
        "class_weight": class_weight,
        "verbose": verbose,
        "max_iter": max_iter,
        "decision_function_shape": decision_function_shape,
        "break_ties": break_ties,
        "random_state": random_state,
    }

    # process data
    X_train, X_test, y_train, y_test = process_data(data, random_state)

    # train model
    model = train_model(params, X_train, X_test, y_train, y_test)

    return Outputs(model_output=model, test_data=(X_test, y_test))


def process_data(df, random_state):
    # split dataframe into X and y
    X = df.drop(["species"], axis=1)
    y = df["species"]

    # train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=random_state
    )

    # return split data
    return X_train, X_test, y_train, y_test


def train_model(params, X_train, X_test, y_train, y_test):
    # train model
    model = SVC(**params)
    model = model.fit(X_train, y_train)

    # return model
    return model