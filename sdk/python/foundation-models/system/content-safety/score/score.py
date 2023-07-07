import json
import logging
import numpy as np
import os

from copy import deepcopy
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
    key = cs_client.accounts.list_keys(resource_group_name=resource_group_name, account_name=aacs_account_name).key1

    return key


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
        sample_input_file_path = os.path.join(model_path, model.saved_input_example_info["artifact_path"])
        with open(sample_input_file_path, "r") as sample_input_file:
            loaded_input = json.load(sample_input_file)
            if model.saved_input_example_info["type"] == "dataframe":
                sample_input = pd.read_json(
                    json.dumps(loaded_input),
                    orient=model.saved_input_example_info["pandas_orient"],
                    dtype=False,
                )
            elif model.saved_input_example_info["type"] == "ndarray":
                inputs = loaded_input["inputs"]
                if isinstance(inputs, dict):
                    sample_input = {input_name: np.asarray(input_value) for input_name, input_value in inputs.items()}
                else:
                    sample_input = np.asarray(inputs)
            else:
                _logger.warning(
                    'Unable to handle sample model input of type "{}", must be of type '
                    '"dataframe" or "ndarray. For more information, please see: '
                    'https://aka.ms/aml-mlflow-deploy."'.format(model.saved_input_example_info["type"])
                )
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
        input_param = PandasParameterType(sample_input, enforce_shape=False, orient="split")

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
        output_param = PandasParameterType(sample_output, enforce_shape=False, orient="records")

# we use mlflow.pyfunc's load_model function because it has a predict function on it we need for inferencing
model = load_model(model_path)


def init():
    # check if aacs access key is provided
    get_aacs_access_key()
    _logger.info("init")


def analyze_text(text):
    endpoint = os.environ.get("CONTENT_SAFETY_ENDPOINT")
    key = get_aacs_access_key()

    # Create an Content Safety client
    client = ContentSafetyClient(endpoint, AzureKeyCredential(key))

    # Build request
    request = AnalyzeTextOptions(text=text)

    # Analyze text
    try:
        response = client.analyze_text(request)
    except Exception as e:
        raise e

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
    if "input_data" in input_data:
        input_data = input_data["input_data"]
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

    result = model.predict(input)
    severity = max([analyze_text(row[0]) for _, row in result.iterrows()])
    if severity > 2:
        return ""

    return _get_jsonable_obj(result, pandas_orient="records")
