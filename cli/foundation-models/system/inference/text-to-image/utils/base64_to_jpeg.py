# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

# Read base64 encoded image from txt file and convert it to image file

import argparse
import re
import json
import io
import base64
from PIL import Image


INPUT_PROMPT_COLUMN = "prompt"
OUTPUT_IMAGE_COLUMN = "generated_image"


def base64_str_to_image(response_file: str):
    """
    Read file that contains response from online endpoint. Extarct base64 encoded images and save them as image files.

    :param response_file: Path to json file, which has response received from endpoint.
    :type response_file: str
    :return: None
    """
    with open(response_file) as f:
        json_str = json.load(f)

    json_obj = json.loads(json_str)
    for i, obj in enumerate(json_obj):
        generated_image = obj[OUTPUT_IMAGE_COLUMN]
        img = Image.open(io.BytesIO(base64.b64decode(generated_image)))
        text_prompt = ""
        if INPUT_PROMPT_COLUMN in obj:
            text_prompt = obj[INPUT_PROMPT_COLUMN].strip()
        text_prompt = f"Img_{i}_" + re.sub(r"[^a-zA-Z0-9 ]+", "", text_prompt)
        text_prompt = text_prompt[:50]
        img.save(text_prompt + ".jpg", "JPEG")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Prepare data for image classification"
    )
    parser.add_argument(
        "--response_file",
        type=str,
        default="generated_image.json",
        help="File having image response from endpoint.",
    )
    args, unknown = parser.parse_known_args()

    base64_str_to_image(args.response_file)
