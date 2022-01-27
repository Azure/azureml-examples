import json
import os
import pandas as pd

from inference_schema.parameter_types.pandas_parameter_type import PandasParameterType
from inference_schema.schema_decorators import input_schema, output_schema
from mlflow.pyfunc import load_model
from mlflow.pyfunc.scoring_server import parse_json_input, _get_jsonable_obj


def init():
    global model

    model_path = os.path.join(os.getenv("AZUREML_MODEL_DIR"), "model")
    model = load_model(model_path)


# Conditional used to determine whether input or output schema decorators are applied
input_schema_condition = "MLFLOW_INPUT_FORMAT_STR" in os.environ
output_schema_condition = "MLFLOW_OUTPUT_FORMAT_STR" in os.environ

# The defaults here from os.getenv should not be invoked, included to avoid parsing error
sample_input = pd.read_json(os.getenv("MLFLOW_INPUT_FORMAT_STR", "{}"), orient="split")
sample_output = pd.read_json(os.getenv("MLFLOW_OUTPUT_FORMAT_STR", "{}"), orient="records")

# Wrapper for applying a decorator conditionally
def conditional_decorator(decorator, condition):
    def wrapper(function):
        if condition:
            return decorator(function)
        else:
            return function

    return wrapper


@conditional_decorator(
    input_schema("input_data", PandasParameterType(sample_input, enforce_column_type=False, orient="split")),
    input_schema_condition,
)
@conditional_decorator(output_schema(PandasParameterType(sample_output, orient="records")), output_schema_condition)
def run(input_data):
    if input_schema_condition or output_schema_condition:
        return _get_jsonable_obj(model.predict(input_data), pandas_orient="records")
    else:
        input_data = json.loads(input_data)
        input_data = input_data["input_data"]
        input_df = parse_json_input(json_input=json.dumps(input_data), orient="split")
        return _get_jsonable_obj(model.predict(input_df), pandas_orient="records")
