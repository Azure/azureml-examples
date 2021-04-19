import argparse 
import pandas as pd

from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, log_loss
import xgboost as xgb
from azureml.core import Workspace, Run

import mlflow
import mlflow.xgboost


def parse_args():
    parser = argparse.ArgumentParser(description="XGBoost example")
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=0.3,
        help="learning rate to update step size at each boosting step (default: 0.3)",
    )
    parser.add_argument(
        "--subsample",
        type=float,
        default=1.0,
        help="subsample ratio of the training instances (default: 1.0)",
    )
    parser.add_argument(
        "--data",
        default=None,
        help="subsample ratio of the training instances (default: 1.0)",
    )
    return parser.parse_args()


def main():
    # parse command-line arguments
    args = parse_args()

    # prepare train and test data
    iris = datasets.load_iris()

    if args.data is not None:
        # Workaround for datastore not fetching data files as expected, only nested dirs
        X = pd.read_csv(args.data + '/iris.csv')
    else:
        X = iris.data
    y = iris.target
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtest = xgb.DMatrix(X_test, label=y_test)

    run = Run.get_context()
    # enable auto logging
    # mlflow.xgboost.autolog()

    # train model
    params = {
        "objective": "multi:softprob",
        "num_class": 3,
        "learning_rate": args.learning_rate,
        "eval_metric": "mlogloss",
        "colsample_bytree": 1,
        "subsample": args.subsample,
        "seed": 42,
    }
    model = xgb.train(params, dtrain, evals=[(dtrain, "train")])

    # evaluate model
    y_proba = model.predict(dtest)
    y_pred = y_proba.argmax(axis=1)
    loss = log_loss(y_test, y_proba)
    acc = accuracy_score(y_test, y_pred)

    # log metrics
    run.log("accuracy", acc)
    run.log("log_loss", loss)

if __name__ == "__main__":
    main()