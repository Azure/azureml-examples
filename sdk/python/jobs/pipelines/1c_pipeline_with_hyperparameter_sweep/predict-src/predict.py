# imports
import os
import mlflow
import argparse
from pathlib import Path
import pandas as pd

from sklearn.svm import SVC
from sklearn.model_selection import train_test_split

parser = argparse.ArgumentParser()

parser.add_argument("--model", type=str)
parser.add_argument("--test_data", type=str)
parser.add_argument("--predict_result", type=str)

args = parser.parse_args()

X_test = pd.read_csv(Path(args.test_data) / "X_test.csv") 
# Fix: Ensure the model path is properly resolved
model_path = str(Path(args.model).resolve())
print(f"Loading model from: {model_path}")
model = mlflow.sklearn.load_model(model_path)
y_test = pd.read_csv(Path(args.test_data) / "y_test.csv")
y_test["predict"] = model.predict(X_test)

y_test.to_csv(Path(args.predict_result) / "predict_result.csv")
