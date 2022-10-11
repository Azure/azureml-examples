import joblib
import os
import pandas as pd
from pathlib import Path
import json 

models = None

def init():
    global models
    model_dir = Path(os.getenv("AZUREML_MODEL_DIR")) / "models"
    models = {m[:-4]: joblib.load(model_dir / m) for m in os.listdir(model_dir)}

def run(data):
    data = json.loads(data)
    model = models[data["model"]] 
    payload = pd.DataFrame(data["data"])
    try: 
        ret = model.predict(payload)
        return pd.DataFrame(ret).to_json()
    except KeyError:
        raise KeyError("No such model")