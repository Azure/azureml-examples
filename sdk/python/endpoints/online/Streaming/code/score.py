from azureml.contrib.services.aml_request import AMLRequest, rawhttp
from azureml.contrib.services.aml_response import AMLResponse
import time, os, logging, joblib, numpy, json


def init():
    global model
    model_path = os.path.join(
        os.getenv("AZUREML_MODEL_DIR"), "model/sklearn_regression_model.pkl"
    )
    model = joblib.load(model_path)
    print("Init complete")


def generate(items):
    for item in items:
        time.sleep(3)
        data = numpy.array(item["data"])
        result = model.predict(data)
        yield json.dumps(result.tolist())


@rawhttp
def run(request: AMLRequest):
    logging.info("model 1: request received")
    data = request.data
    print(request.get_data())
    items = json.loads(data)["items"]
    return AMLResponse(generate(items), 200)
