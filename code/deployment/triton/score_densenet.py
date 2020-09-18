import numpy as np

import io

from azureml.contrib.services.aml_request import rawhttp
from azureml.contrib.services.aml_response import AMLResponse
from PIL import Image
from utils import get_model_info, parse_model_http, triton_init, triton_infer
from tritonclientutils import triton_to_np_dtype


def preprocess(img, scaling, dtype):
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

    npdtype = triton_to_np_dtype(dtype)
    typed = resized.astype(npdtype)

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


def postprocess(results, output_name, batch_size, batching):
    """Post-process results to show the predicted label."""

    output_array = results.as_numpy(output_name)
    if len(output_array) != batch_size:
        raise Exception(
            "expected {} results, got {}".format(batch_size, len(output_array))
        )

    # Include special handling for non-batching models
    output = ""
    for results in output_array:
        if not batching:
            results = [results]
        for result in results:
            if output_array.dtype.type == np.bytes_:
                cls = "".join(chr(x) for x in result).split(":")
            else:
                cls = result.split(":")
            output += "    {} ({}) = {}".format(cls[0], cls[1], cls[2])

    return output


def init():
    triton_init()
    print(get_model_info())


@rawhttp
def run(request):
    """This function is called every time your webservice receives a request.

    Notice you need to know the names and data types of the model inputs and
    outputs. You can get these values by reading the model configuration file
    or by querying the model metadata endpoint.
    """

    if request.method == "POST":
        model_name = "densenet_onnx"
        input_meta, input_config, output_meta, output_config = parse_model_http(
            model_name=model_name
        )

        input_name = input_meta[0]["name"]
        input_dtype = input_meta[0]["datatype"]
        output_name = output_meta[0]["name"]

        reqBody = request.get_data(False)
        img = Image.open(io.BytesIO(reqBody))
        image_data = preprocess(img, scaling="INCEPTION", dtype=input_dtype)

        mapping = {input_name: image_data}

        res = triton_infer(
            model_name=model_name,
            input_mapping=mapping,
            binary_data=True,
            binary_output=True,
            class_count=1,
        )

        result = postprocess(
            results=res, output_name=output_name, batch_size=1, batching=False
        )

        return AMLResponse(result, 200)
    else:
        return AMLResponse("bad request", 500)
