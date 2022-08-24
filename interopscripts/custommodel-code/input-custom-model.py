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
    print("file_arg:", args.file_arg)

    res = []
    for (dir_path, dir_names, file_names) in os.walk(args.file_arg):
        res.extend(file_names)
    print(res)


if __name__ == "__main__":
    main()
