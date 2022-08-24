import argparse
from random import random
import mlflow
from azureml.core import Run
import joblib
import os

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--file_arg", required=True)


    args = parser.parse_args()
    print("validated")
    print("curr dir: ", os.getcwd())
    print("file_arg:", args.file_arg)
    
    ws = Run.get_context().experiment.workspace
    mlflow.set_tracking_uri(ws.get_mlflow_tracking_uri())

    model = mlflow.sklearn.load_model(args.file_arg)
    print (model)


if __name__ == "__main__":
    main()
