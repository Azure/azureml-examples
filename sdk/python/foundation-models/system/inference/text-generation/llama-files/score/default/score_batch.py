import json
import logging
import numpy as np
import os

from mlflow.pyfunc import load_model
from mlflow.pyfunc.scoring_server import _get_jsonable_obj
from azure.ai.contentsafety import ContentSafetyClient
from azure.core.credentials import AzureKeyCredential
from azure.ai.contentsafety.models import AnalyzeTextOptions


_logger = logging.getLogger(__name__)

# Pandas installed, may not be necessary for tensorspec based models, so don't require it all the time
pandas_installed = False
try:
    import pandas as pd

    pandas_installed = True
except ImportError as exception:
    _logger.warning("Unable to import pandas")


class CsChunkingUtils:
    def __init__(self, chunking_n=1000, delimiter="."):
        self.delimiter = delimiter
        self.chunking_n = chunking_n

    def chunkstring(self, string, length):
        return (string[0 + i : length + i] for i in range(0, len(string), length))

    def split_by(self, input):
        max_n = self.chunking_n
        split = [e + self.delimiter for e in input.split(self.delimiter) if e]
        ret = []
        buffer = ""

        for i in split:
            # if a single element > max_n, chunk by max_n
            if len(i) > max_n:
                ret.append(buffer)
                ret.extend(list(self.chunkstring(i, max_n)))
                buffer = ""
                continue
            if len(buffer) + len(i) <= max_n:
                buffer = buffer + i
            else:
                ret.append(buffer)
                buffer = i

        if len(buffer) > 0:
            ret.append(buffer)
        return ret


def init():
    global aacs_client
    endpoint = os.environ.get("CONTENT_SAFETY_ENDPOINT")
    key = os.environ.get("CONTENT_SAFETY_KEY")
    # Create an Content Safety client
    aacs_client = ContentSafetyClient(endpoint, AzureKeyCredential(key))
    global model

    # AZUREML_MODEL_DIR is an environment variable created during deployment
    model_path = os.path.join(os.environ["AZUREML_MODEL_DIR"], "mlflow_model_folder")
    print(f"## Model path is: {model_path} ##")
    print("## Loading model ##")
    model = load_model(model_path)
    print("## Model load is done ##")


def analyze_response(response):
    severity = 0

    print("## Analyze response ##")

    if response.hate_result is not None:
        _logger.info("Hate severity: {}".format(response.hate_result.severity))
        severity = max(severity, response.hate_result.severity)
    if response.self_harm_result is not None:
        _logger.info("SelfHarm severity: {}".format(response.self_harm_result.severity))
        severity = max(severity, response.self_harm_result.severity)
    if response.sexual_result is not None:
        _logger.info("Sexual severity: {}".format(response.sexual_result.severity))
        severity = max(severity, response.sexual_result.severity)
    if response.violence_result is not None:
        _logger.info("Violence severity: {}".format(response.violence_result.severity))
        severity = max(severity, response.violence_result.severity)
    print(f"## Returning severity {severity} ##")
    return severity


def analyze_text(text):
    # Chunk text
    chunking_utils = CsChunkingUtils(chunking_n=1000, delimiter=".")
    split_text = chunking_utils.split_by(text)

    print("## Calling ACS ##")

    severity = [
        analyze_response(aacs_client.analyze_text(AnalyzeTextOptions(text=i)))
        for i in split_text
    ]
    print(f"## Returning MAX from severity list {severity} ##")
    return max(severity)


def iterate(obj):
    if isinstance(obj, dict):
        result = {}
        for key, value in obj.items():
            result[key] = iterate(value)
        return result
    elif isinstance(obj, list):
        return [iterate(item) for item in obj]
    elif isinstance(obj, str):
        if analyze_text(obj) > 2:
            return ""
        else:
            return obj
    else:
        return obj


def get_safe_response(result):
    jsonable_result = _get_jsonable_obj(result, pandas_orient="records")

    print(jsonable_result)
    return iterate(jsonable_result)


def run(mini_batch):
    resultList = []
    print(f"## Mini batch is {mini_batch} ##")
    for file_path in mini_batch:
        print(f"## Handling file at {file_path} ##")
        input_data = pd.read_csv(file_path)
        print(f"## Predicting with  {input_data['text']} ##")
        result = model.predict(input_data["text"])
        print(f"## Prediction result is {result} ##")
        filtered_result = get_safe_response(result)
        print(f"## Adding filtered result {filtered_result} ##")
        resultList.append(filtered_result)
        return resultList
