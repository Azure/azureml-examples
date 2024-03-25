import os
import logging
import json
import numpy
import joblib


def init():
    """
    This function is called when the container is initialized/started, typically after create/update of the deployment.
    You can write the logic here to perform init operations like caching the model in memory
    """
    global model
    # AZUREML_MODEL_DIR is an environment variable created during deployment.
    # It is the path to the model folder (./azureml-models/$MODEL_NAME/$VERSION)
    # Please provide your model's folder name if there is one
    model_path = os.path.join(
        os.getenv("AZUREML_MODEL_DIR"), "sklearn_regression_model.pkl"
    )
    # deserialize the model file back into a sklearn model
    model = joblib.load(model_path)
    
    # Print statements are logged by App Insights along with other STDOUT/STDERR output in the
    # "trace" logs under STDOUT. Within the init function, a series of zeroes will be prepended as follows:
    # `00000000-0000-0000-0000-000000000000,User init function invoked.`
    print("Init complete")


def run(raw_data):
    """
    This function is called for every invocation of the endpoint to perform the actual scoring/prediction.
    In the example we extract the data from the json input and call the scikit-learn model's predict()
    method and return the result back
    """

    # STDOUT/STDERR output within the run function will have the request-id prefix
    # automatically prepended as follows:
    #`04c6f58d-510f-4e3a-933e-60ac20f2707d,User run function invoked.`
    print("model 2: request received")
    result = [0.5, 0.5]
    print("Request processed")
    data = json.loads(raw_data)["data"]
    data = numpy.array(data)
    result = model.predict(data)
    return result.tolist()
