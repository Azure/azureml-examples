from inference_schema.schema_decorators import input_schema, output_schema
from inference_schema.parameter_types.numpy_parameter_type import NumpyParameterType
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
    param_name="iris",
    param_type=NumpyParameterType(np.array([[7.2, 3.2, 6.0, 1.8]]))
)
@output_schema(
    output_type=StandardPythonParameterType({
        "Category" : ["Virginica"]
    })
)
def run(iris):
    logging.info(type(iris))
    probabilities = model.predict_proba(iris)
    predicted_categories = np.argmax(probabilities, axis=1)
    predicted_categories = np.choose(predicted_categories, category_list).flatten()
    result = {
        "Category": predicted_categories.tolist(),
    }
    return result