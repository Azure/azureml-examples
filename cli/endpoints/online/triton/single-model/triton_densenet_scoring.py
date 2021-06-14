import argparse
import numpy as np
import io
import os
import requests
from PIL import Image

import gevent.ssl
import tritonclient.http as tritonhttpclient


def preprocess(img_content):
    """Pre-process an image to meet the size, type and format
    requirements specified by the parameters.
    """
    c = 3
    h = 224
    w = 224

    img = Image.open(io.BytesIO(img_content))

    sample_img = img.convert("RGB")

    resized_img = sample_img.resize((w, h), Image.BILINEAR)
    resized = np.array(resized_img)
    if resized.ndim == 2:
        resized = resized[:, :, np.newaxis]

    typed = resized.astype(np.float32)

    # scale for INCEPTION
    scaled = (typed / 128) - 1

    # Swap to CHW
    ordered = np.transpose(scaled, (2, 0, 1))

    # Channels are in RGB order. Currently model configuration data
    # doesn't provide any information as to other channel orderings
    # (like BGR) so we just assume RGB.
    img_array = np.array(ordered, dtype=np.float32)[None, ...]

    return img_array


def postprocess(max_label):
    """Post-process results to show the predicted label."""

    absolute_path = os.path.abspath(__file__)
    folder_path = os.path.dirname(absolute_path)
    label_path = os.path.join(folder_path, "densenet_labels.txt")
    print(label_path)

    label_file = open(label_path, "r")
    labels = label_file.read().split("\n")
    label_dict = dict(enumerate(labels))
    final_label = label_dict[max_label]
    return f"{max_label} : {final_label}"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_url")
    parser.add_argument("--token")
    parser.add_argument("--image_url", type=str, default="https://aka.ms/peacock-pic")
    args = parser.parse_args()

    scoring_uri = args.base_url[8:]
    triton_client = tritonhttpclient.InferenceServerClient(
        url=scoring_uri,
        ssl=True,
        ssl_context_factory=gevent.ssl._create_default_https_context,
    )

    headers = {}
    headers["Authorization"] = f"Bearer {args.token}"

    # Check status of triton server
    health_ctx = triton_client.is_server_ready(headers=headers)
    print("Is server ready - {}".format(health_ctx))

    # Check status of model
    model_name = "densenet_onnx"
    status_ctx = triton_client.is_model_ready(model_name, "1", headers)
    print("Is model ready - {}".format(status_ctx))

    img_content = requests.get(args.image_url).content
    img_data = preprocess(img_content)

    # Populate inputs and outputs
    input = tritonhttpclient.InferInput("data_0", img_data.shape, "FP32")
    input.set_data_from_numpy(img_data)
    inputs = [input]
    output = tritonhttpclient.InferRequestedOutput("fc6_1")
    outputs = [output]

    result = triton_client.infer(model_name, inputs, outputs=outputs, headers=headers)
    max_label = np.argmax(result.as_numpy("fc6_1"))
    label_name = postprocess(max_label)
    print(label_name)
