# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license.

import os
import traceback
import yaml
import pandas as pd
import numpy as np

from mlflow.pyfunc import load_model
from mlflow.pyfunc.scoring_server import _get_jsonable_obj
from mlflow.types import DataType
from mlflow.types.schema import Schema

from PIL import Image
import logging


def init():
    global g_model
    global g_schema
    global g_dtypes
    global g_logger
    global g_file_loader_dictionary

    g_logger = logging.getLogger("azureml")
    g_logger.setLevel(logging.INFO)

    model_dir = os.environ["AZUREML_MODEL_DIR"]
    model_rootdir = os.listdir(model_dir)[0]
    model_path = os.path.join(model_dir, model_rootdir)

    g_model = load_model(model_path)
    g_schema = get_input_schema(model_path)
    g_dtypes = get_dtypes(g_schema)
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


def get_dtypes(schema):
    try:
        if schema is None:
            return None
        elif schema.is_tensor_spec():
            data_type = schema.numpy_types()[0]
            data_shape = schema.inputs[0].shape
            g_logger.info(f"Enforcing tensor type : {data_type} and shape: {data_shape}")
            return data_type, data_shape
        else:
            column_dtypes = dict(zip(schema.input_names(), schema.pandas_types()))
            g_logger.info("Enforcing datatypes:" + str(column_dtypes))
            return column_dtypes
    except Exception as e:
        raise Exception("Error reading types from model signature schema: " + schema.__str__() + str(e))


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
    if g_schema and g_schema.is_tensor_spec():
        g_logger.info("Reading csv input file into numpy array because the model specifies tensor input signature.")
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
        g_logger.info(f"Loading the input image file into a numpy array of shape {data_shape} and type {data_type}.")

    data_array = _load_image_as_array(data_file)

    if g_schema is None:
        g_logger.warn("Model input signature is not provided.")
        # Default data_shape:
        #   Create a batch of 1 element with the tensor size of the data_file after it is cast to numpy.ndarray.
        #   Use asterisk to unpack the ndarray.shape tuple and prepend the 1.
        data_type, data_shape = np.uint8, (1, *data_array.shape)
    elif g_schema.is_tensor_spec():
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
        data_type = g_schema.input_types()
        if len(data_type) == 1 and data_type[0] == DataType.binary:
            g_logger.info("Loading the input image file into bytes to put within the pd.DataFrame column.")
            return pd.DataFrame({g_schema.input_names()[0]: [_load_image_as_bytes(data_file)]})
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
                    err_message = "Error processing input file: '" + str(data_file) + "'. Exception:" + str(e)
                    g_logger.error(err_message)
                    file_error_messages[
                        baseFilename
                    ] = f"See logs/user/stdout/0/process000.stdout.txt for traceback on error: {str(e)}"
            if len(result) < len(arguments):
                # Error Threshold should be used here.
                printed_errs = "\n".join(
                    [str(filename) + ":\n" + str(error) + "\n" for filename, error in file_error_messages.items()]
                )
                raise Exception(
                    f"Not all input files were processed successfully. Record of errors by filename:\n{printed_errs}"
                )
            if ableToConcatenate:
                return pd.concat(result)
            if attemptToJsonify:
                for idx, model_output in enumerate(result):
                    result[idx] = _get_jsonable_obj(model_output, pandas_orient="records")
            return result

    return wrapper


@input_filelist_decorator
def run(batch_input):
    predict_result = []
    try:
        predict_result = g_model.predict(batch_input)
    except Exception as e:
        g_logger.error("Processing mini batch failed with exception: " + str(e))
        g_logger.error(traceback.format_exc())
        raise e
    return predict_result
