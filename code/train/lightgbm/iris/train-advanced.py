# imports
import os
import time
import mlflow
import argparse

import pandas as pd
import lightgbm as lgb
import matplotlib.pyplot as plt

from sklearn.metrics import log_loss, accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

# define functions
def preprocess_data(df):
    X = df.drop(["species"], axis=1)
    y = df["species"]

    enc = LabelEncoder()
    y = enc.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    return X_train, X_test, y_train, y_test, enc


def train_model(params, num_boost_round, X_train, X_test, y_train, y_test):
    t1 = time.time()
    train_data = lgb.Dataset(X_train, label=y_train)
    test_data = lgb.Dataset(X_test, label=y_test)
    model = lgb.train(
        params,
        train_data,
        num_boost_round=num_boost_round,
        valid_sets=[test_data],
        valid_names=["test"],
    )
    t2 = time.time()

    return model, t2 - t1


def evaluate_model(model, X_test, y_test):
    y_proba = model.predict(X_test)
    y_pred = y_proba.argmax(axis=1)
    loss = log_loss(y_test, y_proba)
    acc = accuracy_score(y_test, y_pred)

    return loss, acc


print("*" * 60)
print("\n\n")

# enable auto logging
mlflow.lightgbm.autolog()

# arg parser
parser = argparse.ArgumentParser()
parser.add_argument("--data-dir", type=str)
parser.add_argument("--num-boost-round", type=int, default=10)
parser.add_argument("--boosting", type=str, default="gbdt")
parser.add_argument("--num-iterations", type=int, default=16)
parser.add_argument("--num-leaves", type=int, default=31)
parser.add_argument("--num-threads", type=int, default=0)
parser.add_argument("--learning-rate", type=float, default=0.1)
parser.add_argument("--metric", type=str, default="multi_logloss")
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--verbose", type=int, default=0)
args = parser.parse_args()

# setup parameters
num_boost_round = args.num_boost_round

params = {
    "objective": "multiclass",
    "num_class": 3,
    "boosting": args.boosting,
    "num_iterations": args.num_iterations,
    "num_leaves": args.num_leaves,
    "num_threads": args.num_threads,
    "learning_rate": args.learning_rate,
    "metric": args.metric,
    "seed": args.seed,
    "verbose": args.verbose,
}

# read in data
df = pd.read_csv(args.data_dir)

# preprocess data
X_train, X_test, y_train, y_test, enc = preprocess_data(df)

# train model
model, train_time = train_model(
    params, num_boost_round, X_train, X_test, y_train, y_test
)
mlflow.log_metric("training_time", train_time)

# evaluate model
loss, acc = evaluate_model(model, X_test, y_test)
mlflow.log_metrics({"loss": loss, "accuracy": acc})

print("\n\n")
print("*" * 60)
