import os
import json
import pickle
import numpy as np


def init():
    global model
    model_dir = os.getenv("AZUREML_MODEL_DIR")
    # Handle possible directory layouts
    model_path = os.path.join(model_dir, "model.pkl")
    if not os.path.exists(model_path):
        model_path = os.path.join(model_dir, "credit_defaults_model", "model.pkl")
    with open(model_path, "rb") as f:
        model = pickle.load(f)


def run(raw_data):
    data = json.loads(raw_data)
    input_data = data.get("input_data", data)
    result = model.predict(np.array(input_data["data"]))
    return json.dumps(result.tolist())
