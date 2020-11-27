import time
import mlflow
import argparse
import lightgbm as lgb
import pandas as pd
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path")
    args = parser.parse_args()

    print("loading data...")
    df = pd.read_csv(args.data_path)
    print(df.head())

    # preprocess data
    print("preprocessing data...")
    X_train, X_test, y_train, y_test, enc = preprocess_data(df)

    # set training parameters
    params = {
        "objective": "multiclass",
        "num_class": 3,
        "learning_rate": 0.1,
        "metric": "multi_logloss",
        "colsample_bytree": 1.0,
        "subsample": 1.0,
        "seed": 42,
    }

    num_boost_round = 32

    # start run
    run = mlflow.start_run()

    # enable automatic logging
    mlflow.lightgbm.autolog()

    # train model
    print("training model...")
    model, train_time = train_model(
        params, num_boost_round, X_train, X_test, y_train, y_test
    )
    mlflow.log_metric("training_time", train_time)

    # evaluate model
    print("evaluating model...")
    loss, acc = evaluate_model(model, X_test, y_test)
    mlflow.log_metrics({"loss": loss, "accuracy": acc})

    # end run
    mlflow.end_run()


if __name__ == "__main__":
    main()
