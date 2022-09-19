# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import numpy as np
import argparse
import os
import glob

import matplotlib.pyplot as plt

import keras
from keras.models import Sequential, model_from_json
from keras.layers import Dense
from keras.optimizers import RMSprop
from keras.callbacks import Callback

import tensorflow as tf

import mlflow
import mlflow.keras

from utils import load_data, one_hot_encode

print("Keras version:", keras.__version__)
print("Tensorflow version:", tf.__version__)

parser = argparse.ArgumentParser()
parser.add_argument('--data-folder', type=str, dest='data_folder', default='data', help='data folder mounting point')
parser.add_argument('--batch-size', type=int, dest='batch_size', default=50, help='mini batch size for training')
parser.add_argument('--first-layer-neurons', type=int, dest='n_hidden_1', default=100,
                    help='# of neurons in the first layer')
parser.add_argument('--second-layer-neurons', type=int, dest='n_hidden_2', default=100,
                    help='# of neurons in the second layer')
parser.add_argument('--learning-rate', type=float, dest='learning_rate', default=0.001, help='learning rate')

args = parser.parse_args()

# Start Logging
mlflow.start_run()

data_folder = args.data_folder
print('Data folder:', data_folder)

# load train and test set into numpy arrays
# note we scale the pixel intensity values to 0-1 (by dividing it with 255.0) so the model can converge faster.
X_train = load_data(glob.glob(os.path.join(data_folder, '**/train-images-idx3-ubyte.gz'),
                              recursive=True)[0], False) / np.float32(255.0)
X_test = load_data(glob.glob(os.path.join(data_folder, '**/t10k-images-idx3-ubyte.gz'),
                             recursive=True)[0], False) / np.float32(255.0)
y_train = load_data(glob.glob(os.path.join(data_folder, '**/train-labels-idx1-ubyte.gz'),
                              recursive=True)[0], True).reshape(-1)
y_test = load_data(glob.glob(os.path.join(data_folder, '**/t10k-labels-idx1-ubyte.gz'),
                             recursive=True)[0], True).reshape(-1)

print(X_train.shape, y_train.shape, X_test.shape, y_test.shape, sep='\n')

training_set_size = X_train.shape[0]

n_inputs = 28 * 28
n_h1 = args.n_hidden_1
n_h2 = args.n_hidden_2
n_outputs = 10
n_epochs = 20
batch_size = args.batch_size
learning_rate = args.learning_rate

y_train = one_hot_encode(y_train, n_outputs)
y_test = one_hot_encode(y_test, n_outputs)
print(X_train.shape, y_train.shape, X_test.shape, y_test.shape, sep='\n')

# Build a simple MLP model
model = Sequential()
# first hidden layer
model.add(Dense(n_h1, activation='relu', input_shape=(n_inputs,)))
# second hidden layer
model.add(Dense(n_h2, activation='relu'))
# output layer
model.add(Dense(n_outputs, activation='softmax'))

model.summary()

model.compile(loss='categorical_crossentropy',
              optimizer=RMSprop(lr=learning_rate),
              metrics=['accuracy'])

class LogRunMetrics(Callback):
    # callback at the end of every epoch
    def on_epoch_end(self, epoch, log):
        # log a value repeated which creates a list
        mlflow.log_metric('Loss', log['val_loss'])
        mlflow.log_metric('Accuracy', log['val_accuracy'])

history = model.fit(X_train, y_train,
                    batch_size=batch_size,
                    epochs=n_epochs,
                    verbose=2,
                    validation_data=(X_test, y_test),
                    callbacks=[LogRunMetrics()])

score = model.evaluate(X_test, y_test, verbose=0)

# log a single value
mlflow.log_metric("Final test loss", score[0])
print('Test loss:', score[0])

mlflow.log_metric('Final test accuracy', score[1])
print('Test accuracy:', score[1])

fig = plt.figure(figsize=(6, 3))
plt.title('MNIST with Keras MLP ({} epochs)'.format(n_epochs), fontsize=14)
plt.plot(history.history['val_accuracy'], 'b-', label='Accuracy', lw=4, alpha=0.5)
plt.plot(history.history['val_loss'], 'r--', label='Loss', lw=4, alpha=0.5)
plt.legend(fontsize=12)
plt.grid(True)

# log an image
mlflow.log_figure(fig, "Accuracy vs Loss.png")

##########################
#<save and register model>
##########################
# Registering the model to the workspace
print("Registering the model via MLFlow")
registered_model_name="keras_dnn_mnist_model"
mlflow.keras.log_model(
    keras_model=model,
    registered_model_name=registered_model_name,
    artifact_path=registered_model_name,
    extra_pip_requirements=["protobuf~=3.20"]
)

# # Saving the model to a file
print("Saving the model via MLFlow")
mlflow.keras.save_model(
    keras_model=model,
    path=os.path.join(registered_model_name, "trained_model"),
    extra_pip_requirements=["protobuf~=3.20"]
)
###########################
#</save and register model>
###########################

mlflow.end_run()