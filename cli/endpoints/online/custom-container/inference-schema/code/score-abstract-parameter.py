import json 
import os
import logging
from this import s
import numpy as np 
import joblib
from inference_schema.schema_decorators import input_schema, output_schema
from inference_schema.parameter_types.numpy_parameter_type import AbstractParameterType
from inference_schema.parameter_types.standard_py_parameter_type import StandardPythonParameterType
from pathlib import Path


def init():
    model_dir = Path(os.getenv("AZUREML_MODEL_DIR")).resolve()
    model_dir = model_dir / "iris"

    global model
    model = joblib.load(model_dir / "iris.pkl")

class MyParameterType(AbstractParameterType):
    
    def __init__(self, sample_input):
        self.sample_input = sample_input
        # Prevent parsing the list as individual types 
        self.sample_data_type = object

    def deserialize_input(self, input_data): 
        return np.array(input_data)

    def input_to_swagger(self):
        swagger = {"title" : "MyParameterType",
                   "example" : self.sample_input}
        return swagger

category_list = np.array(["Setosa", "Versicolor", "Virginica"])
param_input_iris = MyParameterType([1.0, 2.0, 3.0, 4.0])

@input_schema(param_name="iris", param_type=param_input_iris)
def run(iris):
    probabilities = model.predict_proba(iris)
    predicted_categories = np.argmax(probabilities, axis=1)
    predicted_categories = np.choose(predicted_categories, category_list).flatten() 
    result = {"Probabilities" : probabilities.tolist(),
              "Predicted Categories" : predicted_categories.tolist()}
    return result
