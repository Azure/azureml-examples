import json
import numpy as np
import os
import tensorflow as tf

from azureml.core.model import Model


def init():
    global tf_model
    model_root = os.getenv("AZUREML_MODEL_DIR")
    # the name of the folder in which to look for tensorflow model files
    tf_model_folder = "model"

    tf_model = tf.saved_model.load(os.path.join(model_root, tf_model_folder))


def run(raw_data):
    data = np.array(json.loads(raw_data)["data"], dtype=np.float32)

    # make prediction
    out = tf_model(data)
    y_hat = np.argmax(out, axis=1)

    return y_hat.tolist()
