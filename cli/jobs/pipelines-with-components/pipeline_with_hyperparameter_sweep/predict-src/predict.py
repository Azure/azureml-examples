# imports
import os
import mlflow
import argparse
from pathlib import Path
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str)
parser.add_argument("--test_data", type=str)
parser.add_argument("--predict_result", type=str)

args = parser.parse_args()

# ✅ REQUIRED FOR HYPERPARAMETER (SWEEP) JOBS
os.makedirs(args.predict_result, exist_ok=True)

# load data
X_test = pd.read_csv(Path(args.test_data) / "X_test.csv")
y_test = pd.read_csv(Path(args.test_data) / "y_test.csv")

# load model
model = mlflow.sklearn.load_model(args.model)

# predict
y_test["predict"] = model.predict(X_test)

# save results
y_test.to_csv(Path(args.predict_result) / "predict_result.csv", index=False)
