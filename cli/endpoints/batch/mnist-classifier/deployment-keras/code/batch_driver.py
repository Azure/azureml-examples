import os
import numpy as np
import pandas as pd
import tensorflow as tf
from typing import List
from os.path import basename
from PIL import Image
from tensorflow.keras.models import load_model


def init():
    global model

    # AZUREML_MODEL_DIR is an environment variable created during deployment
    model_path = os.path.join(os.environ["AZUREML_MODEL_DIR"], "model")

    # load the model
    model = load_model(model_path)

def run(mini_batch : List[str]) -> pd.DataFrame:
    print(f"Executing run method over batch of {len(mini_batch)} files.")

    results = []
    for image_path in mini_batch:
        data = Image.open(image_path)
        data = np.array(data)
        data_batch = tf.expand_dims(data, axis=0)

        # perform inference
        pred = model.predict(data_batch)

        # Compute probabilities, classes and labels
        pred_prob = tf.math.reduce_max(tf.math.softmax(pred, axis=-1)).numpy()
        pred_class = tf.math.argmax(pred, axis=-1).numpy()

        results.append({
            "file": basename(image_path), 
            "class": pred_class[0],
            "probability": pred_prob
        })

    return pd.DataFrame(results)
