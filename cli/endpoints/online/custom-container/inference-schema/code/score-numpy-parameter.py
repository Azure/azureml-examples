import os
import numpy as np
import joblib
from inference_schema.schema_decorators import input_schema, output_schema
from inference_schema.parameter_types.numpy_parameter_type import NumpyParameterType
from inference_schema.parameter_types.standard_py_parameter_type import (
    StandardPythonParameterType,
)
from pathlib import Path


def init():
    model_dir = Path(os.getenv("AZUREML_MODEL_DIR")).resolve()
    model_dir = model_dir / "models/iris"

    global model
    model = joblib.load(model_dir / "iris.pkl")


category_list = np.array(["Setosa", "Versicolor", "Virginica"])

# Inference Schema ParameterType objects are defined using sample objects
param_input_iris = NumpyParameterType(np.random.random(4)[np.newaxis, ...])
param_output_proba = NumpyParameterType(
    np.random.random(3)[np.newaxis, ...], enforce_column_type=False, enforce_shape=False
)
param_output_cats = NumpyParameterType(category_list)

param_output = StandardPythonParameterType(
    {"Probabilities": param_output_proba, "Predicted Categories": param_output_cats}
)

# Inference Schema schema decorators are applied to the run function
# param_name corresponds to the named run function argument
# input_schema decorators can be stacked to specify multiple named run function arguments
@input_schema(param_name="iris", param_type=param_input_iris)
@output_schema(param_output_cats)
def run(iris):
    probabilities = model.predict_proba(iris)
    predicted_categories = np.argmax(probabilities, axis=1)
    predicted_categories = np.choose(predicted_categories, category_list).flatten()
    result = {
        "Probabilities": probabilities.tolist(),
        "Predicted Categories": predicted_categories.tolist(),
    }
    return result
