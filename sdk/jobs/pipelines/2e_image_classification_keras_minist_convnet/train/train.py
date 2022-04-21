from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.utils import to_categorical
from keras.callbacks import Callback

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import mlflow

# Get input file
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


def train(train_input, model_output, epochs):

    train_file = get_file(train_input)
    data_train = pd.read_csv(train_file, header=None)
    X = np.array(data_train.iloc[:, 1:])
    y = to_categorical(np.array(data_train.iloc[:, 0]))

    img_rows, img_cols = 28, 28
    input_shape = (img_rows, img_cols, 1)

    # Split validation data to optimiza classifier during training
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=13
    )

    X_train = (
        X_train.reshape(X_train.shape[0], img_rows, img_cols, 1).astype("float32") / 255
    )

    X_val = X_val.reshape(X_val.shape[0], img_rows, img_cols, 1).astype("float32") / 255

    batch_size = 256
    num_classes = 10
    epochs = epochs

    # Construct neuron network
    model = Sequential()
    model.add(
        Conv2D(
            32,
            kernel_size=(3, 3),
            activation="relu",
            kernel_initializer="he_normal",
            input_shape=input_shape,
        )
    )
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.25))
    model.add(Conv2D(64, (3, 3), activation="relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Conv2D(128, (3, 3), activation="relu"))
    model.add(Dropout(0.4))
    model.add(Flatten())
    model.add(Dense(128, activation="relu"))
    model.add(Dropout(0.3))
    model.add(Dense(num_classes, activation="softmax"))

    model.compile(
        loss=keras.losses.categorical_crossentropy,
        optimizer=keras.optimizers.Adam(),
        metrics=["accuracy"],
    )

    # Log metrics
    class LogRunMetrics(Callback):
        # callback at the end of every epoch
        def on_epoch_end(self, epoch, log):
            # log a value repeated which creates a list
            mlflow.log_metric("Loss", log["loss"])
            mlflow.log_metric("Accuracy", log["accuracy"])

    history = model.fit(
        X_train,
        y_train,
        batch_size=batch_size,
        epochs=epochs,
        verbose=1,
        validation_data=(X_val, y_val),
        callbacks=[LogRunMetrics()],
    )

    # Log an image for training loss and accuracy
    fig = plt.figure(figsize=(6, 3))
    plt.title("Fashion MNIST with Keras ({} epochs)".format(epochs), fontsize=14)
    plt.plot(history.history["accuracy"], "b-", label="Accuracy", lw=4, alpha=0.5)
    plt.plot(history.history["loss"], "r--", label="Loss", lw=4, alpha=0.5)
    plt.legend(fontsize=12)
    plt.grid(True)
    mlflow.log_figure(fig, "Loss v.s. Accuracy.png")

    # Output model file
    model.save(model_output + "/image_classification_model.h5")
