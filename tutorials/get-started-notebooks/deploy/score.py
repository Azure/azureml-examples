import os
import json
import logging

import mlflow
import pandas as pd

_model = None


def init():
    """Load the MLflow model from AZUREML_MODEL_DIR."""
    global _model
    model_root = os.environ["AZUREML_MODEL_DIR"]
    # AZUREML_MODEL_DIR may point to either the model folder itself or its parent.
    candidate = model_root
    if not os.path.exists(os.path.join(candidate, "MLmodel")):
        # Look one level deeper (registered model layout: <root>/<name>/MLmodel).
        for entry in os.listdir(model_root):
            sub = os.path.join(model_root, entry)
            if os.path.isdir(sub) and os.path.exists(os.path.join(sub, "MLmodel")):
                candidate = sub
                break
    _model = mlflow.pyfunc.load_model(candidate)
    logging.info("Loaded MLflow model from %s", candidate)


def run(raw_data):
    """Score a request payload.

    Accepts the standard MLflow `{"input_data": {...}}` schema (split orient).
    """
    data = json.loads(raw_data) if isinstance(raw_data, (str, bytes)) else raw_data
    payload = data.get("input_data", data)

    if isinstance(payload, dict) and "columns" in payload and "data" in payload:
        df = pd.DataFrame(
            data=payload["data"],
            columns=payload["columns"],
            index=payload.get("index"),
        )
    else:
        df = pd.DataFrame(payload)

    preds = _model.predict(df)
    return preds.tolist() if hasattr(preds, "tolist") else list(preds)
