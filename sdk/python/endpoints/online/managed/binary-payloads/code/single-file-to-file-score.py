from azureml.contrib.services.aml_request import AMLRequest, rawhttp
from azureml.contrib.services.aml_response import AMLResponse
from email.mime.multipart import MIMEMultipart
from email.mime.image import MIMEImage
from PIL import Image
import io

default_resize = (128, 128)


def init():
    pass


@rawhttp
def run(req: AMLRequest):
    try:
        data = req.files.getlist("file")[0]
    except IndexError:
        return AMLResponse("No file uploaded", status_code=422)

    img = Image.open(data.stream)
    img = img.resize(default_resize)

    output = io.BytesIO()
    img.save(output, format="JPEG")
    resp = AMLResponse(message=output.getvalue(), status_code=200)
    resp.mimetype = "image/jpg"

    return resp
