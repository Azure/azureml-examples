import argparse
from pathlib import Path
from uuid import uuid4
from datetime import datetime
import os
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import mlflow

mlflow.sklearn.autolog()

parser = argparse.ArgumentParser("train")
parser.add_argument("--train_data", type=str, help="Path to train data")
parser.add_argument("--test_data", type=str, help="Path to test data")
parser.add_argument("--model_output", type=str, help="Path of output model")
parser.add_argument("--test_split_ratio", type=float, help="ratio of train test split")


args = parser.parse_args()

print("hello training world...")

lines = [
    f"Train data path: {args.train_data}",
    f"Test data path: {args.test_data}",
    f"Model output path: {args.model_output}",
    f"Test split ratio:{args.test_split_ratio}",
]

for line in lines:
    print(line)

print("mounted_path files: ")
arr = os.listdir(args.train_data)
print(arr)

df_list = []
for filename in arr:
    print("reading file: %s ..." % filename)
    with open(os.path.join(args.train_data, filename), "r") as handle:
        # print (handle.read())
        input_df = pd.read_csv((Path(args.train_data) / filename))
        df_list.append(input_df)

train_data = df_list[0]
print(train_data.columns)

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
trainX, testX, trainy, testy = train_test_split(
    X, y, test_size=args.test_split_ratio, random_state=42
)
print(trainX.shape)
print(trainX.columns)

# Train a Linear Regression Model with the train set
model = LinearRegression().fit(trainX, trainy)
print(model.score(trainX, trainy))

mlflow.sklearn.save_model(model, args.model_output)

# test_data = pd.DataFrame(testX, columns = )
testX["cost"] = testy
print(testX.shape)
test_data = testX.to_csv(Path(args.test_data) / "test_data.csv")
