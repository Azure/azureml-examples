import pandas as pd
import tensorflow as tf


class TfClassifier:
    def __init__(self, model_path: str, labels_path: str):
        import numpy as np
        from tensorflow.keras.models import load_model

        self.model = load_model(model_path)
        self.imagenet_labels = np.array(open(labels_path).read().splitlines())

    def predict(self, data):

        preds = self.model.predict(data)

        pred_prob = tf.reduce_max(preds, axis=-1)
        pred_class = tf.argmax(preds, axis=-1)
        pred_label = [self.imagenet_labels[pred] for pred in pred_class]

        return pd.DataFrame(
            {"class": pred_class, "probability": pred_prob, "label": pred_label}
        )


def _load_pyfunc(data_path: str):
    import os

    model_path = os.path.abspath(data_path)
    labels_path = os.path.join(model_path, "ImageNetLabels.txt")

    return TfClassifier(model_path, labels_path)
