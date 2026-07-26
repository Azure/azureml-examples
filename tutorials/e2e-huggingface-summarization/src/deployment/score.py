import os
import logging
import json
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
)

def init():
    """
    This function is called when the container is initialized/started, typically after create/update of the deployment.
    You can write the logic here to perform init operations like caching the model in memory
    """
    global model, tokenizer
    # AZUREML_MODEL_DIR is an environment variable created during deployment.
    # It is the path to the model folder (./azureml-models/$MODEL_NAME/$VERSION)
    model_path = os.path.join(os.getenv("AZUREML_MODEL_DIR"), os.listdir(os.getenv("AZUREML_MODEL_DIR"))[0])
    print("model_path")
    print(os.listdir(model_path))
    model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    print("Init complete")


def run(raw_data):
    global model, tokenizer
    """
    This function is called for every invocation of the endpoint to perform the actual scoring/prediction.
    In the example we extract the data from the json input and call the scikit-learn model's predict()
    method and return the result back
    """
    logging.info("Request received")
    article = json.loads(raw_data)["data"]
    if "t5" in model.config.architectures[0].lower():
        article= "summarize:" + article
    
    inputs = tokenizer(article, return_tensors="pt", max_length=512, truncation=True)
    outputs = model.generate(
        inputs["input_ids"], max_length=150, min_length=40, length_penalty=2.0, num_beams=4, early_stopping=True
    )
    result = tokenizer.decode(outputs[0])
    print(result)
    logging.info("Request processed")
    return result
