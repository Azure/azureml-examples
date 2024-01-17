# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license.

import base64
import os
import traceback
from typing import Dict, List, Set, Union
import yaml
import pandas as pd
import numpy as np

from mlflow.pyfunc import load_model
from mlflow.pyfunc.scoring_server import _get_jsonable_obj
from mlflow.types import DataType
from mlflow.types.schema import Schema

from azure.ai.contentsafety import ContentSafetyClient
from azure.core.credentials import AzureKeyCredential
from azure.ai.contentsafety.models import (
    AnalyzeTextOptions,
    AnalyzeImageOptions,
    ImageData,
    AnalyzeImageResult,
    AnalyzeTextResult,
)

from PIL import Image
import logging


def init():
    global g_model
    global g_schema_input
    global g_schema_output
    global g_dtypes
    global g_logger
    global g_file_loader_dictionary
    global aacs_client
    global aacs_enabled

    g_logger = logging.getLogger("azureml")
    g_logger.setLevel(logging.INFO)

    model_dir = os.environ["AZUREML_MODEL_DIR"]
    model_rootdir = os.listdir(model_dir)[0]
    model_path = os.path.join(model_dir, model_rootdir)

    g_model = load_model(model_path)
    g_schema_input = get_input_schema(model_path)
    g_dtypes = get_dtypes(g_schema_input)
    g_schema_output = get_output_schema(model_path)
    g_file_loader_dictionary = {
        ".csv": load_csv,
        ".png": load_image,
        ".jpg": load_image,
        ".jpeg": load_image,
        ".tiff": load_image,
        ".bmp": load_image,
        ".gif": load_image,
        ".parquet": load_parquet,
        ".pqt": load_parquet,
    }

    endpoint = os.environ.get("CONTENT_SAFETY_ENDPOINT", None)
    key = os.environ.get("CONTENT_SAFETY_KEY", None)
    # Create an Content Safety client
    if endpoint is not None and key is not None:
        aacs_client = ContentSafetyClient(endpoint, AzureKeyCredential(key))
        aacs_enabled = True
    else:
        aacs_enabled = False
        g_logger.warn("Azure AI Content Safety (aacs) is disabled.")


def get_input_schema(model_path):
    try:
        with open(os.path.join(model_path, "MLmodel"), "r") as stream:
            ml_model = yaml.load(stream, Loader=yaml.BaseLoader)
        if "signature" in ml_model:
            if "inputs" in ml_model["signature"]:
                return Schema.from_json(ml_model["signature"]["inputs"])
        g_logger.warn(
            "No signature present in MLmodel file: "
            + ml_model.__str__()
            + "\nSignatures are encouraged, but AzureML will take a "
            + "best effort approach to loading the data files."
        )
    except Exception as e:
        raise Exception("Error reading model signature: " + ml_model.__str__() + str(e))


def get_output_schema(model_path: str) -> Schema:
    """Get output schema from model signature.

    :param model_path: path to model
    :type model_path: str
    :raises Exception: Error reading model signature
    :return: output schema
    :rtype: MLflow.types.Schema
    """
    try:
        with open(os.path.join(model_path, "MLmodel"), "r") as stream:
            ml_model = yaml.load(stream, Loader=yaml.BaseLoader)
        if "signature" in ml_model:
            if "outputs" in ml_model["signature"]:
                return Schema.from_json(ml_model["signature"]["outputs"])
        g_logger.warn(
            "No signature present in MLmodel file: "
            + ml_model.__str__()
            + "\nSignatures are encouraged, but AzureML will take a "
            + "best effort approach to loading the data files."
        )
    except Exception as e:
        raise Exception("Error reading model signature: " + ml_model.__str__() + str(e))


def get_dtypes(schema):
    try:
        if schema is None:
            return None
        elif schema.is_tensor_spec():
            data_type = schema.numpy_types()[0]
            data_shape = schema.inputs[0].shape
            g_logger.info(
                f"Enforcing tensor type : {data_type} and shape: {data_shape}"
            )
            return data_type, data_shape
        else:
            column_dtypes = dict(zip(schema.input_names(), schema.pandas_types()))
            g_logger.info("Enforcing datatypes:" + str(column_dtypes))
            return column_dtypes
    except Exception as e:
        raise Exception(
            "Error reading types from model signature schema: "
            + schema.__str__()
            + str(e)
        )


def _get_data_loader(file_extension):
    loader = g_file_loader_dictionary.get(file_extension)
    if loader is None:
        raise ValueError(
            "File type '" + str(file_extension) + "' is not supported. "
            f"Current accepted filetypes are: {list(g_file_loader_dictionary.keys())}"
        )
    else:
        return loader


def load_data(data_file):
    _, file_extension = os.path.splitext(data_file)
    data_loader = _get_data_loader(file_extension)
    return data_loader(data_file)


def load_csv(data_file):
    if g_schema_input and g_schema_input.is_tensor_spec():
        g_logger.info(
            "Reading csv input file into numpy array because the model specifies tensor input signature."
        )
        return np.genfromtxt(data_file, dtype=g_dtypes)
    else:
        g_logger.info(
            "Reading csv input file into pandas dataframe because the model specifies column based input signature or no signature."
        )
        return pd.read_csv(data_file, engine="c", dtype=g_dtypes)


def _load_image_as_array(data_file):
    """
    Loads in the image file and returns as numpy array.

    Args:
        data_file (string): The mounted path of the image file.

    Returns:
        np.ndarray: The image loaded as an array.
    """
    data = Image.open(data_file)
    return np.array(data)


def _load_image_as_bytes(data_file):
    """
    Return the content of the data file as bytes.

    Args:
        data_file (string): The mounted path of the image file.

    Returns:
        bytes: The contents of the file.
    """
    with open(data_file, "rb") as f:
        return f.read()


def load_image(data_file):
    def loading_message(data_shape, data_type):
        g_logger.info(
            f"Loading the input image file into a numpy array of shape {data_shape} and type {data_type}."
        )

    data_array = _load_image_as_array(data_file)

    if g_schema_input is None:
        g_logger.warn("Model input signature is not provided.")
        # Default data_shape:
        #   Create a batch of 1 element with the tensor size of the data_file after it is cast to numpy.ndarray.
        #   Use asterisk to unpack the ndarray.shape tuple and prepend the 1.
        data_type, data_shape = np.uint8, (1, *data_array.shape)
    elif g_schema_input.is_tensor_spec():
        data_type, data_shape = g_dtypes
        if data_shape == (-1,):
            batchsize = 1
            loading_message(data_shape, data_type)
            # Signature indicates different sized images all in a single batch.
            # Create an array of the given dtype (likely object or "O" from the MLflow infer_signature),
            # and insert the np.uint8 image into the dtype="O" batch array.
            batch = np.empty(batchsize, dtype=data_type)
            batch[0] = data_array
            return batch
    else:
        # Schema is ColSpec. Since binary dtype gets interpretted as 'object'
        # when schema.pandas_types() is called in get_dtypes, we must check the
        # actual mlflow input types for binary.
        data_type = g_schema_input.input_types()
        if len(data_type) == 1 and data_type[0] == DataType.binary:
            g_logger.info(
                "Loading the input image file into bytes to put within the pd.DataFrame column."
            )
            return pd.DataFrame(
                {g_schema_input.input_names()[0]: [_load_image_as_bytes(data_file)]}
            )
        else:
            raise TypeError(
                "Scoring image files with a ColSpec signature containing "
                "more than one input column or a column that is not of dtype "
                "'binary' is not yet supported.\n"
                f"This MLflow signature has {len(data_type)} inputs of types: {data_type}."
            )

    loading_message(data_shape, data_type)
    return data_array.astype(data_type).reshape(data_shape)


def load_parquet(data_file):
    g_logger.info(
        "Reading parquet input file into pandas dataframe.\n"
        "Signature validation will occur in the predict function of the MLflow model."
    )
    return pd.read_parquet(data_file)


def nparray_tolist(array):
    """
    Arrays are not JSONable. They must be converted to lists. If the numpy array is a collection of dtype=object
    values, then the _get_jsonable_obj function doesn't properly send them tolist().
    We have seen a np array of dtype="O" when the array holds nested arrays of different sizes.
    To avoid missing these nested arrays, try to jsonify what is in the object array.

    Args:
        array (np.ndarray): The array to convert to list

    Returns:
        aslist (list): The array as a list.
    """
    aslist = []
    if array.dtype == object:  # nested arrays
        for nested in array:
            aslist.append(_get_jsonable_obj(nested))
    else:
        aslist = _get_jsonable_obj(array)
    return aslist


def format_output(model_output, filename):
    """
    Format the model's output as a dataframe and include a column for filename.
    If the output cannot be converted to a dataframe, return a list that has the filename and the output.

    Args:
        model_output: The object returned from the MLflow model's predict function
        filename (str): The base filename of the MiniBatchItem

    Returns:
        model_output (pd.DataFrame or list): The formatted output from the MLflow model with filename included.
    """
    if isinstance(model_output, dict):
        for key, value in model_output.items():
            if isinstance(value, np.ndarray):
                model_output[key] = nparray_tolist(value)
    elif isinstance(model_output, np.ndarray):
        model_output = nparray_tolist(model_output)
    try:
        model_output = pd.DataFrame(model_output)
        model_output["_azureml_filename"] = filename
    except:
        model_output = [filename, str(model_output)]
    return model_output


# wrapping run function for tabular file (csv) and tensor data inputs (img)
def input_filelist_decorator(run_function):
    """
    A wrapper around the model's predict that loads the files of the mini batch and scores them with the model.

    Args:
        run_function (function): The batchDriver's run(batch_input) function, which calls model.predict(batch_input)

    Returns:
        wrapper (function): Wrapper that returns either pd.DataFrame or list of the model outputs
            mapped to the filename of each MiniBatchItem.
    """

    def wrapper(arguments):
        if isinstance(arguments, pd.DataFrame):
            return [str(r) for r in run_function(arguments)]
        else:
            result = []  # PRS needs this to be a list or a pd.DataFrame.
            file_error_messages = {}
            ableToConcatenate = True
            attemptToJsonify = True
            for data_file in arguments:
                try:
                    baseFilename = os.path.basename(data_file)
                    data = load_data(data_file)
                    output = run_function(data)

                    if isinstance(output, pd.DataFrame) or isinstance(output, dict):
                        # Only dataframes and dicts get the new logic.
                        formatted_output = format_output(output, baseFilename)
                        if isinstance(formatted_output, list):
                            ableToConcatenate = False
                        result.append(formatted_output)
                    else:
                        # For back-compatibility, everything else gets old logic.
                        ableToConcatenate = False
                        attemptToJsonify = False
                        for prediction in output:
                            result.append([baseFilename, str(prediction)])
                except Exception as e:
                    err_message = (
                        "Error processing input file: '"
                        + str(data_file)
                        + "'. Exception:"
                        + str(e)
                    )
                    g_logger.error(err_message)
                    file_error_messages[
                        baseFilename
                    ] = f"See logs/user/stdout/0/process000.stdout.txt for traceback on error: {str(e)}"
            if len(result) < len(arguments):
                # Error Threshold should be used here.
                printed_errs = "\n".join(
                    [
                        str(filename) + ":\n" + str(error) + "\n"
                        for filename, error in file_error_messages.items()
                    ]
                )
                raise Exception(
                    f"Not all input files were processed successfully. Record of errors by filename:\n{printed_errs}"
                )
            if ableToConcatenate:
                return pd.concat(result)
            if attemptToJsonify:
                for idx, model_output in enumerate(result):
                    result[idx] = _get_jsonable_obj(
                        model_output, pandas_orient="records"
                    )
            return result

    return wrapper


@input_filelist_decorator
def run(batch_input):
    predict_result = []
    try:
        aacs_threshold = int(os.environ.get("CONTENT_SAFETY_THRESHOLD", default=1))
        if aacs_enabled:
            blocked_input = analyze_data(
                batch_input, aacs_threshold, blocked_input=None, is_input=True
            )
        predict_result = g_model.predict(batch_input)
        if aacs_enabled:
            _ = analyze_data(
                predict_result,
                aacs_threshold,
                blocked_input=blocked_input,
                is_input=False,
            )
    except Exception as e:
        g_logger.error("Processing mini batch failed with exception: " + str(e))
        g_logger.error(traceback.format_exc())
        raise e
    return predict_result


def analyze_data(
    data_frame: pd.DataFrame,
    aacs_threshold: int,
    blocked_input: Set = None,
    is_input: bool = True,
) -> Set:
    """Analyze and remove un-safe data from data-frame using azure content safety service.

    :param data_frame: model input or output data-frame
    :type data_frame: pd.DataFrame
    :param aacs_threshold: azure ai content safety threshold
    :type aacs_threshold: int
    :param blocked_input: set of blocked row index from input data-frame.
    For input data-frame, pass None as it's not evaluated yet, defaults to None
    :type blocked_input: Set, optional
    :param is_input: True if analysis is for input data-frame else False, defaults to True
    :type is_input: bool, optional
    :raises TypeError: Raise exception if passed data-frame is not of pandas data-frame type.
    :return: set of blocked row index if any cell in the row has un-safe content.
    :rtype: Set
    """
    if not isinstance(data_frame, pd.DataFrame):
        raise TypeError("Only pandas data-frame is supported.")
    columns = data_frame.columns
    output_dir = os.environ.get("AZUREML_BI_OUTPUT_PATH", default="")
    mlflow_schema = g_schema_input.to_dict() if is_input else g_schema_output.to_dict()
    blocked_input = set() if blocked_input is None else blocked_input

    for index, row in data_frame.iterrows():
        for column in columns:
            data_type = _check_data_type_from_model_signature(column, mlflow_schema)
            if data_type is None:
                continue
            blocked_index = True if index in blocked_input else False
            image_path = ""
            if blocked_index is False:
                if data_type == "binary":
                    image_path = os.path.join(output_dir, row[column])
                    if (
                        os.path.isfile(image_path)
                        and analyze_image(image_path) > aacs_threshold
                    ):
                        blocked_index = True
                elif (
                    data_type == "string" and analyze_text(row[column]) > aacs_threshold
                ):
                    blocked_index = True

            if blocked_index:
                if is_input:
                    blocked_input.add(index)
                else:
                    if data_type == "binary":
                        image_path = os.path.join(output_dir, str(row[column]))
                        if os.path.isfile(image_path):
                            os.remove(image_path)
                        data_frame.at[
                            index, column
                        ] = "Blocked By Azure AI Content Safety"
    return blocked_input


def _check_data_type_from_model_signature(column_name: str, schema: Dict = None) -> str:
    """Check column data type from model signature

    :param column_name: column name to find it's type
    :type column_name: str
    :param schema: mlflow signature input or output schema
    :type schema: Dict
    :return: type of column name from model signature else return "str"
    :rtype: str
    """
    if schema is None:
        return "string"

    for item in schema:
        if item["name"] == column_name:
            return item["type"]

    return None


class CsChunkingUtils:
    """Chunking utils for text input"""

    def __init__(self, chunking_n=1000, delimiter="."):
        """Initialize chunking utils

        :param chunking_n: chunking size, defaults to 1000
        :type chunking_n: int, optional
        :param delimiter: delimiter character, defaults to "."
        :type delimiter: str, optional
        """
        self.delimiter = delimiter
        self.chunking_n = chunking_n

    def _chunkstring(self, string, length):
        return (string[0 + i : length + i] for i in range(0, len(string), length))

    def split_by(self, input: str) -> List[str]:
        """Split input by delimiter and chunk by chunking_n

        :param input: text input
        :type input: str
        :return: chunked text
        :rtype: List[str]
        """
        max_n = self.chunking_n
        split = [e + self.delimiter for e in input.split(self.delimiter) if e]
        ret = []
        buffer = ""

        for i in split:
            # if a single element > max_n, chunk by max_n
            if len(i) > max_n:
                ret.append(buffer)
                ret.extend(list(self._chunkstring(i, max_n)))
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


def analyze_text(text: str) -> int:
    """Analyze text severity using azure content safety service

    :param image_in_byte64: image in base64 format
    :type image_in_byte64: str
    :return: maximum severity of all categories. If input type is not text then return 0.
    :rtype: int
    """
    if not isinstance(text, str):
        return 0
    chunking_utils = CsChunkingUtils(chunking_n=1000, delimiter=".")
    split_text = chunking_utils.split_by(text)

    print("## Calling ACS ##")

    severity = [
        analyze_aacs_response(aacs_client.analyze_text(AnalyzeTextOptions(text=i)))
        for i in split_text
    ]
    print(f"## Returning MAX from severity list {severity} ##")
    return max(severity)


def analyze_image(image_path: str) -> int:
    """Analyze image severity using azure content safety service

    :param image_in_byte64: image in base64 format
    :type image_in_byte64: str
    :return: maximum severity of all categories
    :rtype: int
    """
    print("Analyzing image...")
    with open(image_path, "rb") as f:
        image = f.read()
        image_in_byte64 = base64.encodebytes(image).decode("utf-8")

    request = AnalyzeImageOptions(image=ImageData(content=image_in_byte64))
    safety_response = aacs_client.analyze_image(request)
    severity = analyze_aacs_response(safety_response)
    print(f"Image Analyzed, severity {severity}")
    return severity


def analyze_aacs_response(
    response: Union[AnalyzeImageResult, AnalyzeTextResult]
) -> int:
    """Analyze response from azure content safety service.

    :param response: response from azure content safety service
    :type response: Union[AnalyzeImageResult, AnalyzeTextResult]
    :return: maximum severity of all categories
    :rtype: int
    """
    severity = 0

    print("## Analyze response ##")

    if response.hate_result is not None:
        severity = max(severity, response.hate_result.severity)
        class_name = "hate"
    if response.self_harm_result is not None:
        severity = max(severity, response.self_harm_result.severity)
        class_name = "self_harm"
    if response.sexual_result is not None:
        severity = max(severity, response.sexual_result.severity)
        class_name = "sexual"
    if response.violence_result is not None:
        severity = max(severity, response.violence_result.severity)
        class_name = "violence"
    print(f"## Returning severity for {class_name} : {severity} ##")
    return severity
