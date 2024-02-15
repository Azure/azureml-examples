"""Script to develop a machine learning model from input data"""
from argparse import ArgumentParser, Namespace
from distutils.dir_util import copy_tree
from typing import Dict, Union

import mltable
import mlflow
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


# define target column
TARGET = ["DEFAULT_NEXT_MONTH"]

# define categorical feature columns
CATEGORICAL_FEATURES = [
    "SEX",
    "EDUCATION",
    "MARRIAGE",
    "PAY_0",
    "PAY_2",
    "PAY_3",
    "PAY_4",
    "PAY_5",
    "PAY_6",
]

# define numeric feature columns
NUMERIC_FEATURES = [
    "LIMIT_BAL",
    "AGE",
    "BILL_AMT1",
    "BILL_AMT2",
    "BILL_AMT3",
    "BILL_AMT4",
    "BILL_AMT5",
    "BILL_AMT6",
    "PAY_AMT1",
    "PAY_AMT2",
    "PAY_AMT3",
    "PAY_AMT4",
    "PAY_AMT5",
    "PAY_AMT6",
]


def main(args: Namespace) -> None:
    """Develop an sklearn model and use mlflow to log metrics"""
    # enable auto logging
    mlflow.sklearn.autolog()

    # setup parameters
    params = {
        "n_estimators": 200,
        "max_depth": 4,
        "criterion": "gini",
        "random_state": 2024,
    }

    # read data
    tbl = mltable.load(args.training_data)
    df = tbl.to_pandas_dataframe()

    # seperate features and target variables
    X = df[CATEGORICAL_FEATURES + NUMERIC_FEATURES]
    y = df[TARGET]

    # split into train and test sets (80% for training, 20% for testing)
    x_train, x_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # train model
    estimator = make_classifer_pipeline(params)
    estimator.fit(x_train, y_train.values.ravel())

    # calculate evaluation metrics
    y_pred = estimator.predict(x_test)
    validation_accuracy_score = accuracy_score(y_test.values.ravel(), y_pred)
    validation_roc_auc_score = roc_auc_score(y_test.values.ravel(), y_pred)
    validation_f1_score = f1_score(y_test.values.ravel(), y_pred)
    validation_precision_score = precision_score(y_test.values.ravel(), y_pred)
    validation_recall_score = recall_score(y_test.values.ravel(), y_pred)

    # log evaluation metrics
    mlflow.log_metric("validation_accuracy_score", validation_accuracy_score)
    mlflow.log_metric("validation_roc_auc_score", validation_roc_auc_score)
    mlflow.log_metric("validation_f1_score", validation_f1_score)
    mlflow.log_metric("validation_precision_score", validation_precision_score)
    mlflow.log_metric("validation_recall_score", validation_recall_score)

    # save models
    mlflow.sklearn.save_model(estimator, "model")

    # copy model artifact to directory
    to_directory = args.model_output
    copy_tree("model", f"{to_directory}")


def make_classifer_pipeline(params: Dict[str, Union[str, int]]) -> Pipeline:
    """Create sklearn pipeline to apply transforms and a final estimator"""
    # categorical features transformations
    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "ohe",
                OneHotEncoder(
                    handle_unknown="ignore",
                ),
            ),
        ]
    )

    # numeric features transformations
    numeric_transformer = Pipeline(
        steps=[("imputer", SimpleImputer(strategy="median"))]
    )

    # preprocessing pipeline
    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", numeric_transformer, NUMERIC_FEATURES),
            ("categorical", categorical_transformer, CATEGORICAL_FEATURES),
        ]
    )

    # model training pipeline
    classifer_pipeline = Pipeline(
        [
            ("preprocessor", preprocessor),
            ("classifier", RandomForestClassifier(**params, n_jobs=-1)),
        ]
    )

    return classifer_pipeline


def parse_args() -> Namespace:
    """Parse command line arguments"""
    # setup arg parser
    parser = ArgumentParser("train")

    # add arguments
    parser.add_argument("--training_data", type=str)
    parser.add_argument("--model_output", type=str)

    # parse args
    args = parser.parse_args()

    return args


if __name__ == "__main__":
    with mlflow.start_run():
        main(parse_args())
