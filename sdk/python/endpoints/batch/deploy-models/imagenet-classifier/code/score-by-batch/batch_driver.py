import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import load_model


def init():
    global model
    global input_width
    global input_height

    # AZUREML_MODEL_DIR is an environment variable created during deployment
    model_path = os.path.join(os.environ["AZUREML_MODEL_DIR"], "model")

    # load the model
    model = load_model(model_path)
    input_width = 244
    input_height = 244


def decode_img(file_path):
    file = tf.io.read_file(file_path)
    img = tf.io.decode_jpeg(file, channels=3)
    img = tf.image.resize(img, [input_width, input_height])
    return img / 255.0


def run(mini_batch):
    images_ds = tf.data.Dataset.from_tensor_slices(mini_batch)
    images_ds = images_ds.map(decode_img).batch(64)

    # perform inference
    pred = model.predict(images_ds)

    # Compute probabilities, classes and labels
    pred_prob = tf.math.reduce_max(tf.math.softmax(pred, axis=-1)).numpy()
    pred_class = tf.math.argmax(pred, axis=-1).numpy()

    return pd.DataFrame(
        {
            "file": mini_batch,
            "class": pred_class,
            "probability": pred_prob,
        }
    )
