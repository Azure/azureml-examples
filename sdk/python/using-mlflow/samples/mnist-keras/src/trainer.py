# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import tensorflow as tf
import tensorflow.keras as keras
import mlflow.tensorflow

# Enable auto-logging to MLflow to capture Keras metrics.
mlflow.tensorflow.autolog()

# Keras version
print("Keras version:", keras.__version__)
mlflow.log_param("keras", keras.__version__)

mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

model = tf.keras.models.Sequential(
    [
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(10),
    ]
)

predictions = model(x_train[:1]).numpy()
print(predictions)
predictions = tf.nn.softmax(predictions).numpy()
print(predictions)
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
print(loss_fn)

model.compile(optimizer="adam", loss=loss_fn, metrics=["accuracy"])

model.fit(x_train, y_train, epochs=1)
model.evaluate(x_test, y_test, verbose=2)
