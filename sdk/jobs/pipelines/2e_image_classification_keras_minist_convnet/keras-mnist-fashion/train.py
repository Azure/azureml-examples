import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.utils import to_categorical
from keras.callbacks import Callback

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import mlflow

parser = argparse.ArgumentParser("train")
parser.add_argument("--input_path", type=str, help="Input path of dataset")
parser.add_argument("--output_model", type=str, help="Output path of model")

args = parser.parse_args()

dataset = args.input_path
print("input dataset path")
print(dataset)
print("input dataset files: ")
arr = os.listdir(dataset)
print(arr)

# split dataset into train and test set
data_all = pd.DataFrame()
for data in os.listdir(dataset):
    df = pd.read_csv(os.path.join(dataset, data), encoding='utf-8', header=None)
    data_all = data_all.append(df, ignore_index=True)
data_all = data_all.sample(frac=1.0) # shuffle all data
cut_idx = int(round(0.2 * data_all.shape[0]))
data_test, data_train = data_all.iloc[:cut_idx], data_all.iloc[cut_idx:]

img_rows, img_cols = 28, 28
input_shape = (img_rows, img_cols, 1)

X = np.array(data_train.iloc[:, 1:])
y = to_categorical(np.array(data_train.iloc[:, 0]))

# here we split validation data to optimiza classifier during training
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=13)

# test data
X_test = np.array(data_test.iloc[:, 1:])
y_test = to_categorical(np.array(data_test.iloc[:, 0]))

X_train = (
    X_train.reshape(X_train.shape[0], img_rows, img_cols, 1).astype("float32") / 255
)
X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols, 1).astype("float32") / 255
X_val = X_val.reshape(X_val.shape[0], img_rows, img_cols, 1).astype("float32") / 255

batch_size = 256
num_classes = 10
epochs = 10

# construct neuron network
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

score = model.evaluate(X_test, y_test, verbose=0)

# log a single value
mlflow.log_metric("Final test loss", score[0])
print("Test loss:", score[0])

mlflow.log_metric("Final test accuracy", score[1])
print("Test accuracy:", score[1])


fig = plt.figure(figsize=(6, 3))
plt.title("Fashion MNIST with Keras ({} epochs)".format(epochs), fontsize=14)
plt.plot(history.history["accuracy"], "b-", label="Accuracy", lw=4, alpha=0.5)
plt.plot(history.history["loss"], "r--", label="Loss", lw=4, alpha=0.5)
plt.legend(fontsize=12)
plt.grid(True)

# log an image
mlflow.log_figure(fig, "Loss v.s. Accuracy.png")

# files saved in the args.output_model folder are automatically uploaded into run history

os.makedirs(args.output_model, exist_ok=True)

# serialize NN architecture to JSON
model_json = model.to_json()
# save model JSON
(Path(args.output_model) / 'model.txt').write_text(model_json)
# save model weights
model.save_weights(os.path.join(args.output_model, "model.h5"))
print(f"model saved in {args.output_model} folder")
