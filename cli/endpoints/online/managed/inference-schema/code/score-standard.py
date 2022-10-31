from inference_schema.schema_decorators import input_schema, output_schema
from inference_schema.parameter_types.standard_py_parameter_type import StandardPythonParameterType
import joblib, os
import numpy as np
import logging 

model = None
category_list = np.array(["Setosa", "Versicolor", "Virginica"])

def init():
    model_dir = os.getenv("AZUREML_MODEL_DIR", "")
    model_dir = os.path.join(model_dir, "models")
    model_path = os.path.join(model_dir, "iris.pkl")
    
    global model
    model = joblib.load(model_path) 

@input_schema(
    param_name="sepal_length",
    param_type=StandardPythonParameterType([7.2])
)
@input_schema(
    param_name="sepal_width",
    param_type=StandardPythonParameterType([3.2])
)
@input_schema(
    param_name="petal_length",
    param_type=StandardPythonParameterType([6.0])
)
@input_schema(
    param_name="petal_width",
    param_type=StandardPythonParameterType([1.8])
)
@output_schema(
    output_type=StandardPythonParameterType({
        "Category" : ["Virginica"]
    })
)
def run(sepal_length, sepal_width, petal_length, petal_width):
    iris = np.array([sepal_length, sepal_width, petal_length, petal_width]).T     
    probabilities = model.predict_proba(iris)
    predicted_categories = np.argmax(probabilities, axis=1)
    predicted_categories = np.choose(predicted_categories, category_list).flatten()
    result = {
        "Category": predicted_categories.tolist(),
    }
    return result