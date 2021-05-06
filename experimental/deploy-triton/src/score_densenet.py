import io
import numpy as np
import os

from azureml.core import Model
from azureml.contrib.services.aml_request import rawhttp
from azureml.contrib.services.aml_response import AMLResponse
from PIL import Image
from onnxruntimetriton import InferenceSession


def preprocess(img, scaling):  # , dtype):
    """Pre-process an image to meet the size, type and format
    requirements specified by the parameters.
    """
    c = 3
    h = 224
    w = 224
    format = "FORMAT_NCHW"

    if c == 1:
        sample_img = img.convert("L")
    else:
        sample_img = img.convert("RGB")

    resized_img = sample_img.resize((w, h), Image.BILINEAR)
    resized = np.array(resized_img)
    if resized.ndim == 2:
        resized = resized[:, :, np.newaxis]

    # npdtype = triton_to_np_dtype(dtype)
    typed = resized.astype(np.float32)
    # typed = resized

    if scaling == "INCEPTION":
        scaled = (typed / 128) - 1
    elif scaling == "VGG":
        if c == 1:
            scaled = typed - np.asarray((128,), dtype=npdtype)
        else:
            scaled = typed - np.asarray((123, 117, 104), dtype=npdtype)
    else:
        scaled = typed

    # Swap to CHW if necessary
    if format == "FORMAT_NCHW":
        ordered = np.transpose(scaled, (2, 0, 1))
    else:
        ordered = scaled

    # Channels are in RGB order. Currently model configuration data
    # doesn't provide any information as to other channel orderings
    # (like BGR) so we just assume RGB.
    return ordered


def postprocess(output_array):
    """Post-process results to show the predicted label."""

    output_array = output_array[0]
    max_label = np.argmax(output_array)
    final_label = label_dict[max_label]
    return f"{max_label} : {final_label}"


def init():
    global session, label_dict
    session = InferenceSession(path_or_bytes="densenet_onnx")

    model_dir = os.path.join(os.environ["AZUREML_MODEL_DIR"], "models")
    folder_path = os.path.join(model_dir, "triton", "densenet_onnx")
    label_path = os.path.join(
        model_dir, "triton", "densenet_onnx", "densenet_labels.txt"
    )
    label_file = open(label_path, "r")
    labels = label_file.read().split("\n")
    label_dict = dict(enumerate(labels))


@rawhttp
async def run(request):
    """This function is called every time your webservice receives a request.

    Notice you need to know the names and data types of the model inputs and
    outputs. You can get these values by reading the model configuration file
    or by querying the model metadata endpoint.
    """

    if request.method == "POST":
        outputs = []

        for output in session.get_outputs():
            outputs.append(output.name)

        input_name = session.get_inputs()[0].name

        reqBody = await request.get_data()
        img = Image.open(io.BytesIO(reqBody))
        image_data = preprocess(img, scaling="INCEPTION")

        res = session.run(outputs, {input_name: image_data})

        result = postprocess(output_array=res)

        return AMLResponse(result, 200)
    else:
        return AMLResponse("bad request", 500)
