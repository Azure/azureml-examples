from pathlib import Path
import os
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
import pickle
import mlflow

import argparse

parser = argparse.ArgumentParser("train")
parser.add_argument("--training_data", type=str, help="Path to training data")
parser.add_argument("--test_data", type=str, help="Path to test data")
parser.add_argument("--model_output", type=str, help="Path of output model")

args = parser.parse_args()

print("hello training world...")

lines = [
    f"Training data path: {args.training_data}",
    f"Test data path: {args.test_data}",
    f"Model output path: {args.model_output}",
]

for line in lines:
    print(line)

# read raw data from csv to dataframe

print("training data files: ")
arr = os.listdir(args.training_data)
print(arr)

train_data = pd.read_csv((Path(args.training_data) / 'transformed_data.csv'))


# Split the data into input(X) and output(y)
y = train_data["cost"]
# X = train_data.drop(['cost'], axis=1)
X = train_data[
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



# Split the data into train and test sets
trainX, testX, trainy, testy = train_test_split(X, y, test_size=0.3, random_state=42)
# print(trainX.shape)
# print(trainX.columns)



# Train a GBDT Regressor Model with the train set
learning_rate = 0.1
n_estimators = 100
model = GradientBoostingRegressor(learning_rate=learning_rate, n_estimators = n_estimators).fit(trainX, trainy)
print("training set score:", model.score(trainX, trainy))
# log params
mlflow.log_param("learning_rate", learning_rate)
mlflow.log_param("n_estimators", n_estimators)



# Output the model and test data
if not os.path.exists(args.model_output):
    os.mkdir(args.model_output)
pickle.dump(model, open((Path(args.model_output) / "model.sav"), "wb"))
# test_data = pd.DataFrame(testX, columns = )

testX["cost"] = testy
print(testX.shape)
test_data = testX.to_csv((Path(args.test_data) / "test_data.csv"))

