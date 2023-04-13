import argparse
import glob
import os
import mlflow
import pandas as pd
from distutils.util import strtobool
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score
from xgboost import XGBClassifier
from mlflow.models.signature import infer_signature

parser = argparse.ArgumentParser("score")
parser.add_argument("--data_path", type=str, help="Path of input model.")
parser.add_argument("--target_column", type=str, help="The target to predict.")
parser.add_argument("--eval_size", type=float, help="The evaluation proportion size", required=False)
parser.add_argument("--register_best_model", type=lambda x: bool(strtobool(x)), help="If we need to register the best model")
parser.add_argument("--registered_model_name", type=str, required=False, help="Name of the model to be registered")
parser.add_argument("--model", type=str, help="The trained model.")
parser.add_argument("--evaluation_results", type=str, help="Path of the evaluation data.")

args = parser.parse_args()
print(vars(args))

with mlflow.start_run(nested=True):
    # Enable autolog
    # mlflow.sklearn.autolog()
    mlflow.xgboost.autolog(log_models=False)

    # Reading input files in CSV format
    input_files = glob.glob(args.data_path + "/*.csv")
    df = pd.concat(map(pd.read_csv, input_files))

    # Determining test dataset
    if args.eval_size > 0:
        train, test = train_test_split(df, test_size=args.eval_size)
    else:
        train = df
        test = df

    # Computing features and target variables
    train_features = train.drop(columns=[args.target_column])
    train_target = train[args.target_column]

    # Training
    model = XGBClassifier(scale_pos_weight=99)
    model.fit(train_features, train_target)

    # Evaluation
    test_features = test.drop(columns=[args.target_column])
    predictions = model.predict(test_features)
    test["Labels"] = predictions
    test["Probabilities"] = model.predict_proba(test_features)[:,1]
    test.to_csv(os.path.join(args.evaluation_results, "test_predictions.csv"), index=False)

    accuracy = accuracy_score(test[args.target_column], predictions)
    recall = recall_score(test[args.target_column], predictions)
    mlflow.log_metrics({ "accuracy": accuracy, "recall": recall })

    # Model logging
    signature = infer_signature(train_features, predictions)
    mlflow.xgboost.save_model(model, args.model, signature=signature)

    if args.register_best_model:
        mlflow.xgboost.log_model(model, "model", signature=signature, registered_model_name=args.registered_model_name)
    else:
        mlflow.xgboost.log_model(model, "model", signature=signature)
