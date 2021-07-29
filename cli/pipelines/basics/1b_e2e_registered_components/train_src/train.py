import argparse
from pathlib import Path
from uuid import uuid4
from datetime import datetime
import os
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import pickle

parser = argparse.ArgumentParser("train")
parser.add_argument("--training_data", type=str, help="Path to training data")
parser.add_argument("--max_epocs", type=int, help="Max # of epocs for the training")
parser.add_argument("--learning_rate", type=float, help="Learning rate")
parser.add_argument("--learning_rate_schedule", type=str, help="Learning rate schedule")
parser.add_argument("--model_output", type=str, help="Path of output model")

args = parser.parse_args()

print("hello training world...")

lines = [
    f"Training data path: {args.training_data}",
    f"Max epocs: {args.max_epocs}",
    f"Learning rate: {args.learning_rate}",
    f"Learning rate: {args.learning_rate_schedule}",
    f"Model output path: {args.model_output}",
]

for line in lines:
    print(line)

print("mounted_path files: ")
arr = os.listdir(args.training_data)
print(arr)

for filename in arr:
    print("reading file: %s ..." % filename)
    with open(os.path.join(args.training_data, filename), "r") as handle:
        print(handle.read())
        input_df = pd.read_csv((Path(args.training_data) / filename))


# Do the train and save the trained model as a file into the output folder.
input_df.columns = input_df.columns.str.replace('"', "")
print(input_df.describe)
print(input_df)
y = input_df[" 2015"]
X = input_df.drop(["Month", " 2015"], axis=1)
# trainX, testX, trainy, testy = train_test_split(X, y, test_size=0.3, random_state=42)
model = LinearRegression().fit(X, y)


# Here output the fitted model.
curtime = datetime.now().strftime("%b-%d-%Y %H:%M:%S")
pickle.dump(model, open((Path(args.model_output) / "model.sav"), "wb"))
