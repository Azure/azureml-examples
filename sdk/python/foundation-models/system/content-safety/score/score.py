import asyncio
import json
import logging
import numpy as np
import os

from copy import deepcopy
from concurrent.futures import ThreadPoolExecutor
from inference_schema.parameter_types.abstract_parameter_type import AbstractParameterType
from inference_schema.parameter_types.numpy_parameter_type import NumpyParameterType
from inference_schema.parameter_types.standard_py_parameter_type import StandardPythonParameterType
from inference_schema.schema_decorators import input_schema, output_schema
from mlflow.models import Model
from mlflow.pyfunc import load_model
from mlflow.pyfunc.scoring_server import _get_jsonable_obj
from azure.ai.contentsafety import ContentSafetyClient
from azure.core.credentials import AzureKeyCredential
from azure.ai.contentsafety.models import AnalyzeTextOptions
from aiolimiter import AsyncLimiter


_logger = logging.getLogger(__name__)

# Pandas installed, may not be necessary for tensorspec based models, so don't require it all the time
pandas_installed = False
try:
    import pandas as pd
    from inference_schema.parameter_types.pandas_parameter_type import PandasParameterType

    pandas_installed = True
except ImportError as exception:
    _logger.warning('Unable to import pandas')

class AsyncRateLimitedOpsUtils:
    # 1000 requests / 10 seconds. Limiting to 800 request per 10 secods
    # limiting to 1000 concurrent requests
    def __init__(self, ops_count=800, ops_seconds=10, concurrent_ops=1000, thread_max_workers=1000):
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

async_rate_limiter = AsyncRateLimitedOpsUtils()

class CsChunkingUtils:
    def __init__(self, chunking_n=1000, delimiter="."):
        self.delimiter = delimiter
        self.chunking_n = chunking_n
    
    def chunkstring(self, string, length):
        return (string[0+i:length+i] for i in range(0, len(string), length))

    def split_by(self, input):
        max_n = self.chunking_n
        split = [e+self.delimiter for e in input.split(self.delimiter) if e]
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


def create_tensor_spec_sample_io(model_signature_io):
    # Create a sample numpy.ndarray based on shape/type of the tensor info of the model
    io = model_signature_io.inputs
    if not model_signature_io.has_input_names():
        # If the input is not a named tensor, the sample io value that we create will just be a numpy.ndarray
        shape = io[0].shape
        if shape and shape[0] == -1:
            # -1 for first dimension means the input data is batched
            # Create a numpy array with the first dimension of shape as 1 so that inference-schema
            # can correctly generate the swagger sample for the input
            shape = list(deepcopy(shape))
            shape[0] = 1
        sample_io = np.zeros(tuple(shape), dtype=io[0].type)
    else:
        # otherwise, the input is a named tensor, so the sample io value that we create will be
        # Dict[str, numpy.ndarray], which maps input name to a numpy.ndarray of the corresponding size
        sample_io = {}
        for io_val in io:
            shape = io_val.shape
            if shape and shape[0] == -1:
                # -1 for first dimension means the input data is batched
                # Create a numpy array with the first dimension of shape as 1 so that inference-schema
                # can correctly generate the swagger sample for the input
                shape = list(deepcopy(shape))
                shape[0] = 1
            sample_io[io_val.name] = np.zeros(tuple(shape), dtype=io_val.type)
    return sample_io


def create_col_spec_sample_io(model_signature_io):
    # Create a sample pandas.DataFrame based on shape/type of the tensor info of the model
    try:
        columns = model_signature_io.input_names()
    except AttributeError:  # MLflow < 1.24.0
        columns = model_signature_io.column_names()
    types = model_signature_io.pandas_types()
    schema = {}
    for c, t in zip(columns, types):
        schema[c] = t
    df = pd.DataFrame(columns=columns)
    return df.astype(dtype=schema)


model_path = os.path.join(os.getenv("AZUREML_MODEL_DIR"), os.getenv("MLFLOW_MODEL_FOLDER"))

# model loaded here using mlfow.models import Model so we have access to the model signature
model = Model.load(model_path)

is_hfv2 = "hftransformersv2" in model.flavors

sample_input = None
input_param = None
sample_output = None
output_param = None

# If a sample input is provided, load this input and use this as the sample input to create the
# scoring script and inference-schema decorators instead of creating a sample based on just the
# signature information
try:
    if model.saved_input_example_info:
        sample_input_file_path = os.path.join(model_path, model.saved_input_example_info['artifact_path'])
        with open(sample_input_file_path, 'r') as sample_input_file:
            loaded_input = json.load(sample_input_file)
            if model.saved_input_example_info['type'] == 'dataframe':
                sample_input = pd.read_json(
                    json.dumps(loaded_input),
                    orient=model.saved_input_example_info['pandas_orient'],
                    dtype=False
                )
            elif model.saved_input_example_info["type"] == "ndarray":
                inputs = loaded_input["inputs"]
                if isinstance(inputs, dict):
                    sample_input = {
                        input_name: np.asarray(input_value) for input_name, input_value in inputs.items()
                    }
                else:
                    sample_input = np.asarray(inputs)
            else:
                _logger.warning('Unable to handle sample model input of type "{}", must be of type '
                                '"dataframe" or "ndarray. For more information, please see: '
                                'https://aka.ms/aml-mlflow-deploy."'.format(model.saved_input_example_info['type']))
except Exception as e:
    _logger.warning(
        "Failure processing model sample input: {}.\nWill attempt to create sample input based on model signature. "
        "For more information, please see: https://aka.ms/aml-mlflow-deploy.".format(e)
    )

# Handle the signature information to attempt creation of a sample based on signature if no concrete
# sample was provided
model_signature = model.signature
if model_signature:
    model_signature_inputs = model_signature.inputs
    model_signature_outputs = model_signature.outputs
    if model_signature_inputs and sample_input is None:
        if model_signature_inputs.is_tensor_spec():
            sample_input = create_tensor_spec_sample_io(model_signature_inputs)
        else:
            sample_input = create_col_spec_sample_io(model_signature_inputs)

    if model_signature_outputs and sample_output is None:
        if model_signature_outputs.is_tensor_spec():
            sample_output = create_tensor_spec_sample_io(model_signature_outputs)
        else:
            sample_output = create_col_spec_sample_io(model_signature_outputs)
else:
    _logger.warning(
        "No signature information provided for model. If no sample information was provided with the model "
        "the deployment's swagger will not include input and output schema and typing information."
        "For more information, please see: https://aka.ms/aml-mlflow-deploy."
    )

if sample_input is None:
    input_param = NoSampleParameterType()
else:
    if isinstance(sample_input, np.ndarray):
        # Unnamed tensor input
        input_param = NumpyParameterType(sample_input, enforce_shape=False)
    elif isinstance(sample_input, dict):
        param_arg = {}
        for key, value in sample_input.items():
            param_arg[key] = NumpyParameterType(value, enforce_shape=False)
        input_param = StandardPythonParameterType(param_arg)
    else:
        input_param = PandasParameterType(sample_input, enforce_shape=False, orient='split')

if sample_output is None:
    output_param = NoSampleParameterType()
else:
    if isinstance(sample_output, np.ndarray):
        # Unnamed tensor input
        output_param = NumpyParameterType(sample_output, enforce_shape=False)
    elif isinstance(sample_output, dict):
        param_arg = {}
        for key, value in sample_output.items():
            param_arg[key] = NumpyParameterType(value, enforce_shape=False)
        output_param = StandardPythonParameterType(param_arg)
    else:
        output_param = PandasParameterType(sample_output, enforce_shape=False, orient='records')

# we use mlflow.pyfunc's load_model function because it has a predict function on it we need for inferencing
model = load_model(model_path)


def init():
    global aacs_client
    endpoint = os.environ.get('CONTENT_SAFETY_ENDPOINT')
    key = os.environ.get('CONTENT_SAFETY_KEY')

    # Create an Content Safety client
    aacs_client = ContentSafetyClient(endpoint, AzureKeyCredential(key))

    _logger.info("init")


async def async_analyze_text_task(client, request):
    loop = asyncio.get_event_loop()
    executor = async_rate_limiter.get_executor()
    sem = async_rate_limiter.get_semaphore()
    await sem.acquire()
    async with async_rate_limiter.get_limiter():
        response = await loop.run_in_executor(executor, client.analyze_text, request)
        sem.release()
        severity = analyze_response(response)
        return severity

def analyze_response(response):
    severity = 0

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

    return severity

def analyze_text(text):
    # Chunk text
    chunking_utils = CsChunkingUtils(chunking_n=1000, delimiter=".")
    split_text = chunking_utils.split_by(text)

    tasks = []
    for i in split_text:
        request = AnalyzeTextOptions(text = i)
        tasks.append(async_analyze_text_task(aacs_client, request))

    done, pending = asyncio.get_event_loop().run_until_complete(
        asyncio.wait(tasks, timeout=60)
    )

    if len(pending) > 0:
        # not all task finished, assume failed
        return 2
    
    return max([d.result() for d in done])

@input_schema("input_data", input_param)
@output_schema(output_param)
def run(input_data):
    if (
        isinstance(input_data, np.ndarray)
        or (isinstance(input_data, dict) and input_data and isinstance(list(input_data.values())[0], np.ndarray))
        or (pandas_installed and isinstance(input_data, pd.DataFrame))
    ):

        result = model.predict(input_data)

        return _get_jsonable_obj(result, pandas_orient="records")

    # Format input
    if isinstance(input_data, str):
        input_data = json.loads(input_data)
    if 'input_data' in input_data:
        input_data = input_data['input_data']
    if is_hfv2:
        input = input_data
    elif isinstance(input_data, list):
        # if a list, assume the input is a numpy array
        input = np.asarray(input_data)
    elif isinstance(input_data, dict) and "columns" in input_data and "index" in input_data and "data" in input_data:
        # if the dictionary follows pandas split column format, deserialize into a pandas Dataframe
        input = pd.read_json(json.dumps(input_data), orient="split", dtype=False)
    else:
        # otherwise, assume input is a named tensor, and deserialize into a dict[str, numpy.ndarray]
        input = {input_name: np.asarray(input_value) for input_name, input_value in input_data.items()}

    # input json string should look something like this: 
    """{
        "input_data": {
            "any": {
                "a_column": "Hi this is example one",
                "b_column": "Hi this is example two"
            }
        }
    }"""

    # note, each additional separate sample would take about 15s to process. Given timeout of 90s, samples > 5 have high chance of timeout.
    result = model.predict(input)
    jsonable_result = _get_jsonable_obj(result, pandas_orient="records")
    # jsnoable result is a list that looks like this:
    """
    [{0: 'first result'}, {0: 'second result'}]
    """

    return_result = []
    for d in jsonable_result:
        result = d[0]
        if analyze_text(result) > 2:
            return_result.append({0: ''})
        else:
            return_result.append(d)


    return return_result