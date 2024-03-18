# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license.

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
from azure.ai.mlmonitoring import Collector
from mlflow.types.utils import _infer_schema
from mlflow.exceptions import MlflowException
from azure.ai.contentsafety import ContentSafetyClient
from azure.core.credentials import AzureKeyCredential
from azure.ai.contentsafety.models import (
    AnalyzeTextOptions,
    AnalyzeImageOptions,
    ImageData,
)
from aiolimiter import AsyncLimiter
from azure.core.pipeline.policies import (
    HeadersPolicy,
)


try:
    aacs_threshold = int(os.environ["CONTENT_SAFETY_THRESHOLD"])
except:
    aacs_threshold = 2


_logger = logging.getLogger(__name__)

# Pandas library might not be required for tensorspec based models, so don't require it all the time
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
    # 1000 requests / 10 seconds. Limiting to 800 request per 10 seconds
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


async_rate_limiter = AsyncRateLimitedOpsUtils()


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


def create_other_sample_io(model_signature_io):
    return model_signature_io


model_path = os.path.join(
    os.getenv("AZUREML_MODEL_DIR"), os.getenv("MLFLOW_MODEL_FOLDER")
)

# model loaded here using mlfow.models import Model so we have access to the model signature
model = Model.load(model_path)

is_hfv2 = "hftransformersv2" in model.flavors
is_transformers = "transformers" in model.flavors
is_langchain = "langchain" in model.flavors
is_openai = "openai" in model.flavors

sample_input = None
input_param = None
sample_output = None
output_param = None


def get_sample_input_from_loaded_example(input_example_info, loaded_input):
    orient = "split" if "columns" in loaded_input else "values"
    if input_example_info["type"] == "dataframe":
        sample_input = pd.read_json(
            json.dumps(loaded_input),
            # needs open source fix
            # orient=input_example_info['pandas_orient'],
            orient=orient,
            dtype=False,
        )
    elif input_example_info["type"] == "ndarray":
        inputs = loaded_input["inputs"]
        if isinstance(inputs, dict):
            sample_input = {
                input_name: np.asarray(input_value)
                for input_name, input_value in inputs.items()
            }
        else:
            sample_input = np.asarray(inputs)
    else:
        # currently unused, as type always comes through from MLflow _Example creation as ndarray or dataframe
        sample_input = loaded_input
        _logger.warning(
            'Potentially unable to handle sample model input of type "{}". The type must be one '
            "of the list detailed in the MLflow repository: "
            "https://github.com/mlflow/mlflow/blob/master/mlflow/types/utils.py#L91 "
            '"dataframe" or "ndarray" is guaranteed to work best. For more information, please see: '
            'https://aka.ms/aml-mlflow-deploy."'.format(
                model.saved_input_example_info["type"]
            )
        )
    return sample_input


# If a sample input is provided, load this input and use this as the sample input to create the
# scoring script and inference-schema decorators instead of creating a sample based on just the
# signature information
try:
    if model.saved_input_example_info:
        sample_input_file_path = os.path.join(
            model_path, model.saved_input_example_info["artifact_path"]
        )
        with open(sample_input_file_path, "r") as sample_input_file:
            loaded_input = json.load(sample_input_file)
            sample_input = get_sample_input_from_loaded_example(
                model.saved_input_example_info, loaded_input
            )
except Exception as e:
    _logger.warning(
        "Failure processing model sample input: {}.\nWill attempt to create sample input based on model signature. "
        "For more information, please see: https://aka.ms/aml-mlflow-deploy.".format(e)
    )


def get_samples_from_signature(
    model_signature_x, previous_sample_input=None, previous_sample_output=None
):
    if model_signature_x is None:
        return previous_sample_input, previous_sample_output
    model_signature_inputs = model_signature_x.inputs
    model_signature_outputs = model_signature_x.outputs
    if model_signature_inputs and previous_sample_input is None:
        if model_signature_inputs.is_tensor_spec():
            sample_input_x = create_tensor_spec_sample_io(model_signature_inputs)
        else:
            try:
                sample_input_x = create_col_spec_sample_io(model_signature_inputs)
            except:
                sample_input_x = create_other_sample_io(model_signature_inputs)
                _logger.warning(
                    "Sample input could not be parsed as either TensorSpec"
                    " or ColSpec. Falling back to taking the sample as is rather than"
                    " converting to numpy arrays or DataFrame."
                )
    else:
        sample_input_x = previous_sample_input

    if model_signature_outputs and previous_sample_output is None:
        if model_signature_outputs.is_tensor_spec():
            sample_output_x = create_tensor_spec_sample_io(model_signature_outputs)
        else:
            sample_output_x = create_col_spec_sample_io(model_signature_outputs)
    else:
        sample_output_x = previous_sample_output
    return sample_input_x, sample_output_x


# Handle the signature information to attempt creation of a sample based on signature if no concrete
# sample was provided
model_signature = model.signature
if model_signature:
    sample_input, sample_output = get_samples_from_signature(
        model_signature, sample_input, sample_output
    )
else:
    _logger.warning(
        "No signature information provided for model. If no sample information was provided with the model "
        "the deployment's swagger will not include input and output schema and typing information."
        "For more information, please see: https://aka.ms/aml-mlflow-deploy."
    )


def get_parameter_type(sample_input_ex, sample_output_ex=None):
    if sample_input_ex is None:
        input_param = NoSampleParameterType()
    else:
        try:
            schema = _infer_schema(sample_input_ex)
            schema_types = schema.input_types
        except MlflowException:
            pass
        finally:
            if isinstance(sample_input_ex, np.ndarray):
                # Unnamed tensor input
                input_param = NumpyParameterType(sample_input_ex, enforce_shape=False)
            elif pandas_installed and isinstance(sample_input_ex, pd.DataFrame):
                # TODO check with OSS about pd.Series
                input_param = PandasParameterType(
                    sample_input_ex, enforce_shape=False, orient="split"
                )
            # elif schema_types and isinstance(sample_input_ex, dict) and not all(stype == DataType.string for stype in schema_types) and \
            #     all(isinstance(value, list) for value in sample_input_ex.values()):
            #     # for dictionaries where there is any non-string type, named tensor
            #     param_arg = {}
            #     for key, value in sample_input_ex.items():
            #         param_arg[key] = NumpyParameterType(value, enforce_shape=False)
            #     input_param = StandardPythonParameterType(param_arg)
            elif isinstance(sample_input_ex, dict):
                # TODO keeping this around while _infer_schema doesn't work on dataframe string signatures
                param_arg = {}
                for key, value in sample_input_ex.items():
                    param_arg[key] = NumpyParameterType(value, enforce_shape=False)
                input_param = StandardPythonParameterType(param_arg)
            else:
                # strings, bytes, lists and dictionaries with only strings as base type
                input_param = NoSampleParameterType()

    if sample_output_ex is None:
        output_param = NoSampleParameterType()
    else:
        if isinstance(sample_output_ex, np.ndarray):
            # Unnamed tensor input
            output_param = NumpyParameterType(sample_output_ex, enforce_shape=False)
        elif isinstance(sample_output_ex, dict):
            param_arg = {}
            for key, value in sample_output_ex.items():
                param_arg[key] = NumpyParameterType(value, enforce_shape=False)
            output_param = StandardPythonParameterType(param_arg)
        else:
            output_param = PandasParameterType(
                sample_output_ex, enforce_shape=False, orient="records"
            )

    return input_param, output_param


input_param, output_param = get_parameter_type(sample_input, sample_output)

# we use mlflow.pyfunc's load_model function because it has a predict function on it we need for inferencing
model = load_model(model_path)


def get_aacs_access_key():
    key = os.environ.get("CONTENT_SAFETY_KEY")

    if key:
        return key

    uai_client_id = os.environ.get("UAI_CLIENT_ID")
    if not uai_client_id:
        raise RuntimeError(
            "Cannot get AACS access key, both UAI_CLIENT_ID and CONTENT_SAFETY_KEY are not set, exiting..."
        )

    subscription_id = os.environ.get("SUBSCRIPTION_ID")
    resource_group_name = os.environ.get("RESOURCE_GROUP_NAME")
    aacs_account_name = os.environ.get("CONTENT_SAFETY_ACCOUNT_NAME")
    from azure.mgmt.cognitiveservices import CognitiveServicesManagementClient
    from azure.identity import ManagedIdentityCredential

    credential = ManagedIdentityCredential(client_id=uai_client_id)
    cs_client = CognitiveServicesManagementClient(credential, subscription_id)
    key = cs_client.accounts.list_keys(
        resource_group_name=resource_group_name, account_name=aacs_account_name
    ).key1

    return key


def init():
    global inputs_collector, outputs_collector, aacs_client
    endpoint = os.environ.get("CONTENT_SAFETY_ENDPOINT")
    key = get_aacs_access_key()

    # Create an Content Safety client
    headers_policy = HeadersPolicy()
    headers_policy.add_header("ms-azure-ai-sender", "text-to-image")
    aacs_client = ContentSafetyClient(
        endpoint, AzureKeyCredential(key), headers_policy=headers_policy
    )

    try:
        inputs_collector = Collector(name="model_inputs")
        outputs_collector = Collector(name="model_outputs")
        _logger.info("Input and output collector initialized")
    except Exception as e:
        _logger.error(
            "Error initializing model_inputs collector and model_outputs collector. {}".format(
                e
            )
        )


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
        print("Hate severity: {}".format(response.hate_result.severity))
        severity = max(severity, response.hate_result.severity)
    if response.self_harm_result is not None:
        print("SelfHarm severity: {}".format(response.self_harm_result.severity))
        severity = max(severity, response.self_harm_result.severity)
    if response.sexual_result is not None:
        print("Sexual severity: {}".format(response.sexual_result.severity))
        severity = max(severity, response.sexual_result.severity)
    if response.violence_result is not None:
        print("Violence severity: {}".format(response.violence_result.severity))
        severity = max(severity, response.violence_result.severity)

    return severity


def analyze_text_async(text):
    # Chunk text
    chunking_utils = CsChunkingUtils(chunking_n=1000, delimiter=".")
    split_text = chunking_utils.split_by(text)

    tasks = []
    for i in split_text:
        request = AnalyzeTextOptions(text=i)
        tasks.append(async_analyze_text_task(aacs_client, request))

    done, pending = asyncio.get_event_loop().run_until_complete(
        asyncio.wait(tasks, timeout=60)
    )

    if len(pending) > 0:
        # not all task finished, assume failed
        return 6

    return max([d.result() for d in done])


def analyze_text(text):
    # Chunk text
    print("Analyzing ...")
    if (not text) or (not text.strip()):
        return 0
    chunking_utils = CsChunkingUtils(chunking_n=1000, delimiter=".")
    split_text = chunking_utils.split_by(text)

    result = [
        analyze_response(aacs_client.analyze_text(AnalyzeTextOptions(text=i)))
        for i in split_text
    ]
    severity = max(result)
    print(f"Analyzed, severity {severity}")

    return severity


def analyze_image(image_in_byte64: str) -> int:
    """Analyze image severity using azure content safety service

    :param image_in_byte64: image in base64 format
    :type image_in_byte64: str
    :return: maximum severity of all categories
    :rtype: int
    """
    print("Analyzing image...")
    if image_in_byte64 is None:
        return 0
    request = AnalyzeImageOptions(image=ImageData(content=image_in_byte64))
    safety_response = aacs_client.analyze_image(request)
    severity = analyze_response(safety_response)
    print(f"Image Analyzed, severity {severity}")
    return severity


def _check_data_type_from_model_signature(key: str) -> str:
    """Check key data type from model signature

    :param key: key of data (to analyze by AACS) in model input or output
    :type key: str
    :return: data type of key from model signature else return "str"
    :rtype: str
    """
    if model_signature is None or key is None:
        return "str"
    input_schema = model_signature.inputs.to_dict()
    output_schema = model_signature.outputs.to_dict()

    def _get_type(schema):
        for item in schema:
            if item["name"] == key:
                return item["type"]
        return None

    return _get_type(input_schema) or _get_type(output_schema) or "str"


def iterate(obj, current_key=None):
    if isinstance(obj, dict):
        severity = 0
        for key, value in obj.items():
            obj[key], value_severity = iterate(value, key)
            severity = max(severity, value_severity)
        return obj, severity
    elif isinstance(obj, list) or isinstance(obj, np.ndarray):
        severity = 0
        for idx in range(len(obj)):
            obj[idx], value_severity = iterate(obj[idx])
            severity = max(severity, value_severity)
        return obj, severity
    elif isinstance(obj, pd.DataFrame):
        severity = 0
        columns = obj.columns
        for i in range(obj.shape[0]):  # iterate over rows
            for j in range(obj.shape[1]):  # iterate over columns
                obj.iloc[i][columns[j]], value_severity = iterate(
                    obj.iloc[i][columns[j]], columns[j]
                )
                severity = max(severity, value_severity)
        return obj, severity
    elif isinstance(obj, str):
        if _check_data_type_from_model_signature(current_key) == "binary":
            severity = analyze_image(obj)
        else:
            severity = analyze_text(obj)
        if severity >= aacs_threshold:
            return "", severity
        else:
            return obj, severity
    else:
        return obj, 0


def get_safe_response(result):
    print("Analyzing response...")
    jsonable_result = _get_jsonable_obj(result, pandas_orient="records")
    result, severity = iterate(jsonable_result)
    print(f"Response analyzed, severity {severity}")
    return result


def get_safe_input(input_data):
    print("Analyzing input...")
    result, severity = iterate(input_data)
    print(f"Input analyzed, severity {severity}")
    return result, severity


@input_schema("input_data", input_param)
@output_schema(output_param)
def run(input_data):
    context = None
    input_data, severity = get_safe_input(input_data)
    if severity > aacs_threshold:
        return {}
    if (
        isinstance(input_data, np.ndarray)
        or (
            isinstance(input_data, dict)
            and input_data
            and isinstance(list(input_data.values())[0], np.ndarray)
        )
        or (pandas_installed and isinstance(input_data, pd.DataFrame))
    ):
        # Collect model input
        try:
            context = inputs_collector.collect(input_data)
        except Exception as e:
            _logger.error(
                "Error collecting model_inputs collection request. {}".format(e)
            )

        result = model.predict(input_data)

        # Collect model output
        try:
            mdc_output_df = pd.DataFrame(result)
            outputs_collector.collect(mdc_output_df, context)
        except Exception as e:
            _logger.error(
                "Error collecting model_outputs collection request. {}".format(e)
            )

        return get_safe_response(result)

        # Collect model input
    try:
        context = inputs_collector.collect(input)
    except Exception as e:
        _logger.error("Error collecting model_inputs collection request. {}".format(e))

    if is_transformers or is_langchain or is_openai:
        input = parse_model_input_from_input_data_transformers(input_data)
    else:
        input = parse_model_input_from_input_data_traditional(input_data)
    result = model.predict(input)

    # Collect output data
    try:
        mdc_output_df = pd.DataFrame(result)
        outputs_collector.collect(mdc_output_df, context)
    except Exception as e:
        _logger.error("Error collecting model_outputs collection request. {}".format(e))

    return get_safe_response(result)


def parse_model_input_from_input_data_traditional(input_data):
    # Format input
    if isinstance(input_data, str):
        input_data = json.loads(input_data)
    if "input_data" in input_data:
        input_data = input_data["input_data"]
    if is_hfv2:
        input = input_data
    elif isinstance(input_data, list):
        # if a list, assume the input is a numpy array
        input = np.asarray(input_data)
    elif (
        isinstance(input_data, dict)
        and "columns" in input_data
        and "index" in input_data
        and "data" in input_data
    ):
        # if the dictionary follows pandas split column format, deserialize into a pandas Dataframe
        input = pd.read_json(json.dumps(input_data), orient="split", dtype=False)
    else:
        # otherwise, assume input is a named tensor, and deserialize into a dict[str, numpy.ndarray]
        input = {
            input_name: np.asarray(input_value)
            for input_name, input_value in input_data.items()
        }
    return input


def parse_model_input_from_input_data_transformers(input_data):
    # Format input
    if isinstance(input_data, str):
        try:
            input_data = json.loads(input_data)
        except ValueError:
            # allow non-json strings to go through
            input = input_data

    if isinstance(input_data, dict) and "input_data" in input_data:
        input_data = input_data["input_data"]

    if is_hfv2:
        input = input_data
    elif isinstance(input_data, str) or isinstance(input_data, bytes):
        # strings and bytes go through
        input = input_data
    elif isinstance(input_data, list) and all(
        isinstance(element, str) for element in input_data
    ):
        # lists of strings go through
        input = input_data
    elif isinstance(input_data, list) and all(
        isinstance(element, dict) for element in input_data
    ):
        # lists of dicts of [str: str | List[str]] go through
        try:
            for dict_input in input_data:
                _validate_input_dictionary_contains_only_strings_and_lists_of_strings(
                    dict_input
                )
            input = input_data
        except MlflowException:
            _logger.error(
                "Could not parse model input - passed a list of dictionaries which had entries which were not strings or lists."
            )
    elif isinstance(input_data, list):
        # if a list, assume the input is a numpy array
        input = np.asarray(input_data)
    elif (
        isinstance(input_data, dict)
        and "columns" in input_data
        and "index" in input_data
        and "data" in input_data
    ):
        # if the dictionary follows pandas split column format, deserialize into a pandas Dataframe
        input = pd.read_json(json.dumps(input_data), orient="split", dtype=False)
    elif isinstance(input_data, dict):
        # if input is a dictionary, but is not all ndarrays and is not pandas, it must only contain strings
        try:
            _validate_input_dictionary_contains_only_strings_and_lists_of_strings(
                input_data
            )
            input = input_data
        except MlflowException:
            # otherwise, assume input is a named tensor, and deserialize into a dict[str, numpy.ndarray]
            input = {
                input_name: np.asarray(input_value)
                for input_name, input_value in input_data.items()
            }
    else:
        input = input_data

    return input


def _validate_input_dictionary_contains_only_strings_and_lists_of_strings(data):
    invalid_keys = []
    invalid_values = []
    value_type = None
    for key, value in data.items():
        if not value_type:
            value_type = type(value)
        if isinstance(key, bool):
            invalid_keys.append(key)
        elif not isinstance(key, (str, int)):
            invalid_keys.append(key)
        if isinstance(value, list) and not all(
            isinstance(item, (str, bytes)) for item in value
        ):
            invalid_values.append(key)
        elif not isinstance(value, (np.ndarray, list, str, bytes)):
            invalid_values.append(key)
        elif isinstance(value, np.ndarray) or value_type == np.ndarray:
            if not isinstance(value, value_type):
                invalid_values.append(key)
    if invalid_values:
        from mlflow.protos.databricks_pb2 import INVALID_PARAMETER_VALUE

        raise MlflowException(
            "Invalid values in dictionary. If passing a dictionary containing strings, all "
            "values must be either strings or lists of strings. If passing a dictionary containing "
            "numeric values, the data must be enclosed in a numpy.ndarray. The following keys "
            f"in the input dictionary are invalid: {invalid_values}",
            error_code=INVALID_PARAMETER_VALUE,
        )
    if invalid_keys:
        raise MlflowException(
            f"The dictionary keys are not all strings or indexes. Invalid keys: {invalid_keys}"
        )
