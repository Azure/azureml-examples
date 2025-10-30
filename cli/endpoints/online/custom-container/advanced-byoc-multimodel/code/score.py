from handler import Handler
from azureml.contrib.services.aml_request import rawhttp
import json 
from PIL import Image
from io import BytesIO
import logging 
import os 

def init(): 
    global handler
    handler = Handler(os.getenv("AZUREML_MODEL_DIR"))

@rawhttp 
def run(request):
    model = request.headers.get("Model")
    payload = json.loads(request.json)
    res = handler.infer(model, payload['value'])
    return res
        
    
