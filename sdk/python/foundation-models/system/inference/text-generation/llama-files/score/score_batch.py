import asyncio
import json
import logging
import numpy as np
import os
from copy import deepcopy
from concurrent.futures import ThreadPoolExecutor
from inference_schema.parameter_types.abstract_parameter_type import (
    AbstractParameterType,
)
from inference_schema.parameter_types.numpy_parameter_type import NumpyParameterType
from inference_schema.parameter_types.standard_py_parameter_type import (
    StandardPythonParameterType,
)
from inference_schema.schema_decorators import input_schema, output_schema
from mlflow.models import Model
from mlflow.pyfunc import load_model
from mlflow.pyfunc.scoring_server import _get_jsonable_obj
from mlflow.types.utils import _infer_schema
from mlflow.exceptions import MlflowException
from azure.ai.contentsafety import ContentSafetyClient
from azure.core.credentials import AzureKeyCredential
from azure.ai.contentsafety.models import AnalyzeTextOptions



_logger = logging.getLogger(__name__)

# Pandas installed, may not be necessary for tensorspec based models, so don't require it all the time
pandas_installed = False
try:
    import pandas as pd
    from inference_schema.parameter_types.pandas_parameter_type import (
        PandasParameterType,
    )

    pandas_installed = True
except ImportError as exception:
    _logger.warning("Unable to import pandas")


class AsyncRateLimitedOpsUtils:
    # 1000 requests / 10 seconds. Limiting to 800 request per 10 secods
    # limiting to 1000 concurrent requests
    def __init__(
        self,
        ops_count=800,
        ops_seconds=10,
        concurrent_ops=1000,
        thread_max_workers=1000,
    ):
        self.limiter = AsyncLimiter(ops_count, ops_seconds)
        self.semaphore = asyncio.Semaphore(value=concurrent_ops)
        # need thread pool executor for sync function
        self.executor = ThreadPoolExecutor(max_workers=thread_max_workers)

    def get_limiter(self):
        return self.limiter

    def get_semaphore(self):
        return self.semaphore

    def get_executor(self):
        return self.executor


#async_rate_limiter = AsyncRateLimitedOpsUtils()


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


class NoSampleParameterType(AbstractParameterType):
    def __init__(self):
        super(NoSampleParameterType, self).__init__(None)

    def deserialize_input(self, input_data):
        """
        Passthrough, do nothing to the incoming data
        """
        return input_data

    def input_to_swagger(self):
        """
        Return schema for an empty object
        """
        return {"type": "object", "example": {}}






def init():
    global aacs_client
    endpoint = os.environ.get("CONTENT_SAFETY_ENDPOINT")
    key = os.environ.get("CONTENT_SAFETY_KEY")
    # print("Key")
    # print(key)
    # print(endpoint)
    # Create an Content Safety client
    aacs_client = ContentSafetyClient(endpoint, AzureKeyCredential(key))
    global model
    

    # AZUREML_MODEL_DIR is an environment variable created during deployment
    print(os.environ["AZUREML_MODEL_DIR"])
    import subprocess
    model_path = os.path.join(os.environ["AZUREML_MODEL_DIR"], "mlflow_model_folder")
    subprocess.run("find  $model_path -type f", shell=True, stdout=subprocess.PIPE)
    print("model path is: ", model_path)
    print("loading model")
    model = load_model(model_path)
    print("model load is done")
    # model_path = os.path.join(
    # os.getenv("AZUREML_MODEL_DIR"), os.getenv("MLFLOW_MODEL_FOLDER")
#)



def analyze_response(response):
    severity = 0

    print("analyze response")

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
    print("Returning severity ", severity)
    return severity


def analyze_text(text):
    # Chunk text
    chunking_utils = CsChunkingUtils(chunking_n=1000, delimiter=".")
    split_text = chunking_utils.split_by(text)

    print("analyze_text")

    severity = [analyze_response(aacs_client.analyze_text(AnalyzeTextOptions(text=i))) for i in split_text]
    print("Returning MAX severity ", severity)
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
    for file_path in mini_batch:
        print("Inside mini_batch, with path, ", file_path)
        input_data = pd.read_csv(file_path)
        print("Doing predict with :", file_path)
        result = model.predict(input_data["text"])
        print("calling ACS")
        return get_safe_response(result)

