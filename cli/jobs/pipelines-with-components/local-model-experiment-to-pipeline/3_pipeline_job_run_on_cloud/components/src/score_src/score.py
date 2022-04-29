from pathlib import Path
import os
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score
import mlflow
import argparse

parser = argparse.ArgumentParser("score")
parser.add_argument(
    "--predictions", type=str, help="Path of predictions and actual data"
)

args = parser.parse_args()

# read raw data from csv to dataframe
print("prediction and actual data files: ")
arr = os.listdir(args.predictions)
print(arr)

test_data = pd.read_csv((Path(args.predictions) / 'predictions.csv'))


lines = [
    f"Predictions path: {args.predictions}",
]

for line in lines:
    print(line)

actuals = test_data["actual_cost"]
predictions = test_data["predicted_cost"]


mlflow.log_metric("mean_squared_error", mean_squared_error(actuals, predictions))
mlflow.log_metric("r2_score", r2_score(actuals, predictions))