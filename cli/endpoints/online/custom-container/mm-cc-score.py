import joblib
from pathlib import Path
import os
import random
import json
import pandas as pd 

def init():
    global models
    global binom_split 
    
    binom_split = os.getenv("binom_split")
    binom_split = 0.2 if not binom_split else float(binom_split)

    model_dir = Path(os.getenv("AZUREML_MODEL_DIR")) / "models" 
    models = {"Champion" : joblib.load(model_dir /  "lasso-alpha-0.1.sav"), 
              "Challenger" : joblib.load(model_dir / "lasso-alpha-0.5.sav")}

def run(data):
    selected_model = "Challenger" if random.random() < binom_split else "Champion"
    mod = models[selected_model]
    data = pd.read_json(data)
    res = mod.predict(data).tolist()
    return json.dumps({"Model" : selected_model, "Result" : res})