import argparse
import pandas as pd
import mlflow.sklearn
import os

parser = argparse.ArgumentParser()
parser.add_argument("--input_model", type=str)
parser.add_argument("--custom_model_output", type=str)
args = parser.parse_args()

sk_model = mlflow.sklearn.load_model(args.input_model)

mlflow.sklearn.save_model(sk_model, os.path.join(args.custom_model_output, "my_model"))
