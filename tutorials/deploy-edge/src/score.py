import tensorflow as tf
import numpy as np
import json
import os
from azureml.contrib.services.aml_request import AMLRequest, rawhttp
from azureml.contrib.services.aml_response import AMLResponse


def init():
    global session
    global input_name
    global output_name

    session = tf.Session()

    # AZUREML_MODEL_DIR is an environment variable created during deployment.
    # It is the path to the model folder (./azureml-models/$MODEL_NAME/$VERSION)
    # For multiple models, it points to the folder containing all deployed models (./azureml-models)
    model_path = os.path.join(os.getenv("AZUREML_MODEL_DIR"), "resnet50")
    model = tf.saved_model.loader.load(session, ["serve"], model_path)
    if len(model.signature_def["serving_default"].inputs) > 1:
        raise ValueError("This score.py only supports one input")
    input_name = [
        tensor.name for tensor in model.signature_def["serving_default"].inputs.values()
    ][0]
    output_name = [
        tensor.name
        for tensor in model.signature_def["serving_default"].outputs.values()
    ]


@rawhttp
def run(request):
    if request.method == "POST":
        reqBody = request.get_data(False)
        resp = score(reqBody)
        return AMLResponse(resp, 200)
    if request.method == "GET":
        respBody = str.encode("GET is not supported")
        return AMLResponse(respBody, 405)
    return AMLResponse("bad request", 500)


def score(data):
    result = session.run(output_name, {input_name: [data]})
    return json.dumps(result[1].tolist())


if __name__ == "__main__":
    init()
    with open("test_image.jpg", "rb") as f:
        content = f.read()
        print(score(content))
