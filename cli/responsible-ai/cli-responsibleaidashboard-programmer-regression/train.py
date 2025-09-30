# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import argparse
import json
import os
import tempfile

from sklearn.ensemble import RandomForestRegressor
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder


import mlflow
import mlflow.sklearn

import mltable

import time


def parse_args():
    # setup arg parser
    parser = argparse.ArgumentParser()

    # add arguments
    parser.add_argument("--training_data", type=str, help="Path to training data")
    parser.add_argument("--target_column_name", type=str, help="Name of target column")
    parser.add_argument("--continuous_features", type=json.loads)
    parser.add_argument("--categorical_features", type=json.loads)
    parser.add_argument("--model_name", type=str, help="model name")
    parser.add_argument("--model_output", type=str, help="model output path")
    parser.add_argument("--model_output_json", type=str, help="model info output path")

    # parse args
    args = parser.parse_args()

    # return args
    return args


def get_regression_model_pipeline(continuous_features, categorical_features):
    # We create the preprocessing pipelines for both numeric and
    # categorical data.
    numeric_transformer = Pipeline(steps=[("scaler", StandardScaler())])

    categorical_transformer = Pipeline(
        steps=[("onehot", OneHotEncoder(handle_unknown="ignore"))]
    )

    transformations = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, continuous_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )

    # Append classifier to preprocessing pipeline.
    # Now we have a full prediction pipeline.
    pipeline = Pipeline(
        steps=[
            ("preprocessor", transformations),
            ("regressor", RandomForestRegressor()),
        ]
    )
    return pipeline


def main(args):
    tracking_uri = mlflow.get_tracking_uri()
    print("tracking_uri: {0}".format(tracking_uri))

    # Read in data
    print("Reading data")
    train_dataset = mltable.load(args.training_data).to_pandas_dataframe()

    # Drop the labeled column to get the training set.
    y_train = train_dataset[args.target_column_name]
    X_train = train_dataset.drop(columns=[args.target_column_name])

    continuous_features = args.continuous_features
    categorical_features = args.categorical_features

    pipeline = get_regression_model_pipeline(
        continuous_features=continuous_features,
        categorical_features=categorical_features,
    )
    model = pipeline.fit(X_train, y_train)

    # Saving model with mlflow
    print("Saving model with MLFlow to model_output directory")
    mlflow.sklearn.save_model(sk_model=model, path=args.model_output)

    suffix = int(time.time())
    registered_name = "{0}_{1}".format(args.model_name, suffix)
    print(f"Registering model as {registered_name}")

    # Use save_model and create_model_version with file:// URI to let Azure ML handle the upload
    with tempfile.TemporaryDirectory() as temp_dir:
        model_dir = os.path.join(temp_dir, registered_name)
        mlflow.sklearn.save_model(model, model_dir)

        # Use the older model registry API directly to avoid logged-models search
        from mlflow.tracking import MlflowClient

        client = MlflowClient()

        try:
            # Try to create the registered model (will fail if it already exists)
            client.create_registered_model(registered_name)
            print(f"Created new registered model: {registered_name}")
        except Exception as e:
            print(f"Registered model {registered_name} already exists: {e}")

        # Create a new version of the model using file:// URI
        # Azure ML will handle the upload and generate the proper azureml:// URI
        file_uri = f"file://{model_dir}"
        print("Registering model with file_uri: {0}".format(file_uri))

        model_version = client.create_model_version(
            name=registered_name, source=file_uri
        )
        print(f"Created model version {model_version.version} for {registered_name}")

    model_info = {"id": "{0}:1".format(registered_name)}
    output_path = os.path.join(args.model_output_json, "model_info.json")
    with open(output_path, "w") as of:
        json.dump(model_info, fp=of)


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
