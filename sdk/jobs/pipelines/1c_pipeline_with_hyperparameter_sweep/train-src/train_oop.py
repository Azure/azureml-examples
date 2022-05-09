# imports
import os
import mlflow
import argparse

import pandas as pd
from pathlib import Path

from sklearn.svm import SVC
from sklearn.model_selection import train_test_split

from azure.ml import Input, Output
from azure.ml.dsl import command_component

@command_component
def main(data: Input,
         model_output: Output, test_data: Output, 
         C=1.0, kernel='rbf', degree=3, gamma='scale', coef0:float=0, 
         shrinking=False, probability=False, tol=1e-3, cache_size=1024, class_weight:bool=None, 
         verbose=False, max_iter=-1, decision_function_shape='ovr', break_ties=False, random_state=42):
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

    # read in data
    df = pd.read_csv(data)

    # process data
    X_train, X_test, y_train, y_test = process_data(df, random_state)

    # train model
    model = train_model(params, X_train, X_test, y_train, y_test)
    # Output the model and test data
    mlflow.sklearn.save_model(model, model_output + "/model")
    X_test.to_csv(Path(test_data) / "X_test.csv", index=False)
    y_test.to_csv(Path(test_data) / "y_test.csv", index=False)


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