from datetime import datetime
import tensorflow as tf
from tensorflow import keras
import time
import numpy as np

data_size = 1000
# 80% of the data is for training.
train_pct = 0.8

train_size = int(data_size * train_pct)

# Create some input data between -1 and 1 and randomize it.
x = np.linspace(-1, 1, data_size)
np.random.shuffle(x)

# Generate the output data.
# y = 0.5x + 2 + noise
y = 0.5 * x + 2 + np.random.normal(0, 0.05, (data_size,))

# Split into test and train pairs.
x_train, y_train = x[:train_size], y[:train_size]
x_test, y_test = x[train_size:], y[train_size:]

logdir = "outputs/tblogs/" + datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = keras.callbacks.TensorBoard(log_dir=logdir)

model = keras.models.Sequential(
    [
        keras.layers.Dense(16, input_dim=1),
        keras.layers.Dense(1),
    ]
)

model.compile(
    loss="mse",  # keras.losses.mean_squared_error
    optimizer=keras.optimizers.SGD(learning_rate=0.2),
)

import datetime

ct = datetime.datetime.now()
print("current time:-", ct)

print("Training ... With default parameters, this takes less than 10 seconds.")
training_history = model.fit(
    x_train,
    y_train,
    batch_size=train_size,
    verbose=0,
    epochs=10000,
    validation_data=(x_test, y_test),
    callbacks=[tensorboard_callback],
)

ct = datetime.datetime.now()
print("current time:-", ct)

print("Average test loss: ", np.average(training_history.history["loss"]))
