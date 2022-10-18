
class TfClassifier():
    def __init__(self, model_path: str, labels_path: str):
        import tensorflow as tf
        import numpy as np
        from tensorflow.keras.models import load_model
        
        self.model = load_model(model_path)
        self.imagenet_labels = np.array(open(labels_path).read().splitlines())

    def predict(self, data):
        results = self.model.predict(data)

        return [self.imagenet_labels[pred] for pred in results]

def _load_pyfunc(data_path: str):
    import os

    model_path = os.path.abspath(data_path)
    labels_path = os.path.join(model_path, 'ImageNetLabels.txt')

    return TfClassifier(model_path, labels_path)
