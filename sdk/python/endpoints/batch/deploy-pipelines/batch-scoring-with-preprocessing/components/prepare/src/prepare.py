import os
import argparse
import glob
from pathlib import Path
from typing import List

import pandas as pd
import numpy as np
import joblib
import sklearn
import mlflow
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer

transform_filename = 'column_transformer.pkl'
continuous_features = ["age", "chol", "oldpeak", "thalach", "trestbps"]
discrete_features = ["ca", "cp", "exang",
                     "fbs", "restecg", "sex", "slope", "thal"]
target_column = "target"


def build_preprocessing_pipeline(categorical_encoding: str, continuous_features: List[str], discrete_features: List[str]) -> ColumnTransformer:
    # Configure the categorical encoder
    if categorical_encoding == "ordinal":
        categorical_encoder = OrdinalEncoder(
            handle_unknown='use_encoded_value', unknown_value=np.nan)
    elif categorical_encoding == "onehot":
        categorical_encoder = OneHotEncoder(handle_unknown='ignore')
    else:
        raise ValueError(
            f"categorical_encoding '{categorical_encoding}' is not a valid encoding strategy. Possible values are 'ordinal' or 'onehot'")

    # For continuous variables, replace missing values with the median and then normalize by
    # subtracting the mean and dividing by the standard deviation
    continuous_pipeline = sklearn.pipeline.Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ]
    )

    # For discrete variables, encode the data
    discrete_pipeline = sklearn.pipeline.Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", categorical_encoder)
        ]
    )

    # Build the pipeline
    transformations = ColumnTransformer(
        [
            ('continuous_pipe', continuous_pipeline, continuous_features),
            ('discrete_pipe', discrete_pipeline, discrete_features)
        ],
        remainder='passthrough')  # Target will passthrough if there

    return transformations


def preprocess_heart_disease_data(df, continuous_features, discrete_features, target, categorical_encoding: str = "ordinal", transformations=None):
    mlflow.sklearn.autolog()

    if target in df.columns:
        features_df = df.drop(columns=[target])
        restore_target = True
    else:
        features_df = df
        restore_target = False

    if transformations:
        df_transformed = transformations.transform(features_df)
    else:
        transformations = build_preprocessing_pipeline(
            categorical_encoding, continuous_features, discrete_features)
        df_transformed = transformations.fit_transform(features_df)

    # Get columns names from transformations
    transformed_discrete_features = transformations.transformers_[1][1].named_steps['encoder'].get_feature_names_out(discrete_features)
    all_features = continuous_features + list(transformed_discrete_features)

    if restore_target:
        target_values = df[target].to_numpy().reshape(len(df), 1)
        df_transformed = np.hstack((df_transformed, target_values))
        all_features.append(target)

    return pd.DataFrame(df_transformed, columns=all_features), transformations


if __name__ == '__main__':
    parser = argparse.ArgumentParser("prepare")
    parser.add_argument("--data_path", type=str,
                        help="Path to the data to transform")
    parser.add_argument("--categorical_encoding", type=str,
                        help="Categorical encoding strategy", default="ordinal", required=False)
    parser.add_argument("--transformations_path", type=str,
                        help="Path of transformations", required=False)
    parser.add_argument("--transformations_output_path",
                        type=str, help="Transformations output path if learned")
    parser.add_argument("--prepared_data_path", type=str, help="Prepared data")

    args = parser.parse_args()

    lines = [
        f"Input data path: {args.data_path}",
        f"Categorical encoding strategy: {args.categorical_encoding}",
        f"transformations path: {args.transformations_path}",
        f"Processed data path: {args.prepared_data_path}",
    ]

    for line in lines:
        print(line)

    print("[DEBUG]Loading transformation if available")
    if args.transformations_path:
        transformations_input_path = os.path.join(
            args.transformations_path, transform_filename)
        if os.path.exists(transformations_input_path):
            print(
                f"[DEBUG]Transformations loaded from {transformations_input_path}")
            transformations = joblib.load(transformations_input_path)
        else:
            print(
                f"[WARN]There are no transformations available at path {transformations_input_path} with the expected name {transform_filename}")
            transformations = None
    else:
        print(f"[INFO]Transformations will be learnt in the preprocessing step")
        transformations = None

    print(f"[DEBUG]Reading all the CSV files from path {args.data_path}")
    arr = os.listdir(args.data_path)
    print(arr)
    file_paths = glob.glob(args.data_path + "/*.csv")
    print("[DEBUG]CSV files:", file_paths)

    with mlflow.start_run(nested=True):
        for file_path in file_paths:
            print(f"[DEBUG]Working with file {file_path}")
            df = pd.read_csv(file_path)
            preprocessed, transformations = preprocess_heart_disease_data(
                df,
                continuous_features,
                discrete_features,
                target_column,
                args.categorical_encoding,
                transformations
            )

            output_file_name = Path(file_path).stem
            output_file_path = os.path.join(
                args.prepared_data_path, output_file_name + '.csv')
            print(f"[DEBUG]Writing file {output_file_path}")
            preprocessed.to_csv(output_file_path, index=False)

    transformations_output_path = os.path.join(
        args.transformations_output_path, transform_filename)
    joblib.dump(transformations, transformations_output_path)
