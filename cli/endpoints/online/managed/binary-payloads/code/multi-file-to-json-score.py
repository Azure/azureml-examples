from azureml.contrib.services.aml_request import AMLRequest, rawhttp
from PIL import Image


def init():
    pass

@rawhttp
def run(req: AMLRequest):
    sizes = [
        {"filename": f.filename, "size": Image.open(f.stream).size}
        for f in req.files.getlist("file[]")
    ]

    return {"response": sizes}
