# This is a sample code for illustration purpose only. Do not use it for production use.
import argparse
import os
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
import pickle
import shutil

print("here")
parser = argparse.ArgumentParser("batch_inference")
parser.add_argument("--inference_data", type=str, help="Path to inference data")
parser.add_argument("--model_input", type=str, help="Path of input model")
parser.add_argument("--output_data", type=str, help="Path of output data")

args = parser.parse_args()


inference_df = pd.read_parquet(os.path.join(args.inference_data, "data"))

categorical_feature_names = ["transactionID", "accountID"]
ordinal_feature_names = [
    "isProxyIP",
    # this feature is from feature store "isUserRegistered",
]

df = inference_df.drop(
    categorical_feature_names + ordinal_feature_names + ["timestamp"], axis="columns"
).fillna(0)

X = df.to_numpy()

# load the model
print("check model path")

with open(args.model_input + "/clf.pkl", "rb") as pickle_file:
    loaded_model = pickle.load(pickle_file)

result = loaded_model.predict(X)

df["predict_is_fraud"] = result

print("output path")
print(args.output_data)

df.to_parquet(args.output_data + "/output.parquet")
print("Inference data saved")
