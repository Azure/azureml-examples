import argparse
import pandas as pd
import os
from pathlib import Path
from sklearn.linear_model import LinearRegression
import mlflow

mlflow.sklearn.autolog()


parser = argparse.ArgumentParser("predict")
parser.add_argument("--model_input", type=str, help="Path of input model")
parser.add_argument("--test_data", type=str, help="Path to test data")
parser.add_argument("--predictions", type=str, help="Path of predictions")

args = parser.parse_args()

print("hello scoring world...")

lines = [
    f"Model path: {args.model_input}",
    f"Test data path: {args.test_data}",
    f"Predictions path: {args.predictions}",
]

for line in lines:
    print(line)

# Load and split the test data

print("mounted_path files: ")
arr = os.listdir(args.test_data)

print(arr)
test_data = pd.read_csv(Path(args.test_data) / "test_data.csv")
testy = test_data["cost"]
# testX = test_data.drop(['cost'], axis=1)
testX = test_data[
    [
        "distance",
        "dropoff_latitude",
        "dropoff_longitude",
        "passengers",
        "pickup_latitude",
        "pickup_longitude",
        "store_forward",
        "vendor",
        "pickup_weekday",
        "pickup_month",
        "pickup_monthday",
        "pickup_hour",
        "pickup_minute",
        "pickup_second",
        "dropoff_weekday",
        "dropoff_month",
        "dropoff_monthday",
        "dropoff_hour",
        "dropoff_minute",
        "dropoff_second",
    ]
]
print(testX.shape)
print(testX.columns)

# load mlflow model
model = mlflow.sklearn.load_model(args.model_input)

# Make predictions on testX data and record them in a column named predicted_cost
predictions = model.predict(testX)
testX["predicted_cost"] = predictions
print(testX.shape)

# Compare predictions to actuals (testy)
output_data = pd.DataFrame(testX)
output_data["actual_cost"] = testy


# Save the output data with feature columns, predicted cost, and actual cost in csv file
output_data = output_data.to_csv((Path(args.predictions) / "predictions.csv"))
