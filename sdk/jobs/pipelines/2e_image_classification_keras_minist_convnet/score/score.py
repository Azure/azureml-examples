from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.utils import to_categorical
from keras.callbacks import Callback
from keras.models import load_model

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import mlflow


def get_file(f):

    f = Path(f)
    if f.is_file():
        return f
    else:
        files = list(f.iterdir())
        if len(files) == 1:
            return files[0]
        else:
            raise Exception("********This path contains more than one file*******")


def parse_args():
    # setup argparse
    parser = argparse.ArgumentParser()

    # add arguments
    parser.add_argument(
        "--input_data", type=str, help="path containing data for scoring"
    )
    parser.add_argument(
        "--input_model", type=str, default="./", help="input path for model"
    )

    parser.add_argument(
        "--output_result", type=str, default="./", help="output path for model"
    )

    # parse args
    args = parser.parse_args()

    # return args
    return args


def score(input_data, input_model, output_result):

    test_file = get_file(input_data)
    data_test = pd.read_csv(test_file, header=None)

    img_rows, img_cols = 28, 28
    input_shape = (img_rows, img_cols, 1)

    # Read test data
    X_test = np.array(data_test.iloc[:, 1:])
    y_test = to_categorical(np.array(data_test.iloc[:, 0]))
    X_test = (
        X_test.reshape(X_test.shape[0], img_rows, img_cols, 1).astype("float32") / 255
    )

    # Load model
    files = [f for f in os.listdir(input_model) if f.endswith(".h5")]
    model = load_model(input_model + "/" + files[0])

    # Log metrics of the model
    eval = model.evaluate(X_test, y_test, verbose=0)

    mlflow.log_metric("Final test loss", eval[0])
    print("Test loss:", eval[0])

    mlflow.log_metric("Final test accuracy", eval[1])
    print("Test accuracy:", eval[1])

    # Score model using test data
    y_predict = model.predict(X_test)
    y_result = np.argmax(y_predict, axis=1)

    # Output result
    np.savetxt(output_result + "/predict_result.csv", y_result, delimiter=",")


def main(args):
    score(args.input_data, args.input_model, args.output_result)


# run script
if __name__ == "__main__":
    # parse args
    args = parse_args()

    # call main function
    main(args)
