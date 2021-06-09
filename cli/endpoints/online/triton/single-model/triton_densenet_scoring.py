import argparse
import numpy as np
import io
import requests
from PIL import Image


def preprocess(img_content, scaling):
    """Pre-process an image to meet the size, type and format
    requirements specified by the parameters.
    """
    c = 3
    h = 224
    w = 224
    format = "FORMAT_NCHW"

    img = Image.open(io.BytesIO(img_content))

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
    img_array = np.array(ordered, dtype=np.float32)[None, ...]

    return img_array


def postprocess(max_label, label_path):
    """Post-process results to show the predicted label."""

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

    headers = {}
    headers["Authorization"] = f"Bearer {args.token}"

    # Check status of triton server
    resp = requests.get(f"{args.base_url}/v2/health/ready", headers=headers)
    print(resp.text)

    # Check status of model
    resp = requests.post(f"{args.base_url}/v2/repository/index", headers=headers)
    print(resp.text)

    # Check metadata of model for inference
    resp = requests.get(f"{args.base_url}/v2/models/densenet_onnx", headers=headers)
    print(resp.text)

    img_content = requests.get(args.image_url).content
    img_data = preprocess(img_content, scaling="INCEPTION")

    score_input = (
        '{"inputs":[{"name":"data_0","data":'
        + str(img_data.flatten().tolist())
        + ',"datatype":"FP32","shape":[1,3,224,224]}]}'
    )
    resp = requests.post(
        f"{args.base_url}/v2/models/densenet_onnx/infer",
        data=score_input,
        headers=headers,
    )
    print(resp.text)
