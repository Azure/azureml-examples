from azureml.contrib.services.aml_request import AMLRequest, rawhttp
from PIL import Image

def init():
    pass

@rawhttp
def run(req: AMLRequest):
    files = req.files.getlist("file[]")
    sizes = [
        {"filename": f.filename, "size": Image.open(f.stream).size} for f in files
    ]
    return {"response": sizes}
