# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

# Prepare request payload for the inpainting task

import argparse
import json
import io
import base64
import os
import pandas as pd
from PIL import Image


def read_image(image_path) -> bytes:
    """Reads an image from a file path into a byte array."""
    with open(image_path, "rb") as f:
        return f.read()


def prepare_batch_payload(payload_path: str) -> None:
    """Prepare payload for online deployment.

    :param payload_path: Path to payload csv file.
    :type payload_path: str
    :return: None
    """

    base_image1 = "inpainting_data/images/dog_on_bench.png"
    mask_image1 = "inpainting_data/images/dog_on_bench_mask.png"
    base_image2 = "inpainting_data/images/teapot.png"
    mask_image2 = "inpainting_data/images/teapot_mask.png"

    os.makedirs(payload_path, exist_ok=True)

    input_data = {
        "columns": ["image", "mask_image", "prompt"],
        "data": [
            {
                "image": base64.encodebytes(read_image(base_image1)).decode("utf-8"),
                "mask_image": base64.encodebytes(read_image(mask_image1)).decode(
                    "utf-8"
                ),
                "prompt": "A yellow cat, high resolution, sitting on a park bench",
            },
            {
                "image": base64.encodebytes(read_image(base_image2)).decode("utf-8"),
                "mask_image": base64.encodebytes(read_image(mask_image2)).decode(
                    "utf-8"
                ),
                "prompt": "A small flower featuring a blend of pink and purple colors.",
            },
        ],
    }
    pd.DataFrame(**input_data).to_csv(
        os.path.join(payload_path, "input1.csv"), index=False
    )

    input_data = {
        "columns": ["image", "mask_image", "prompt"],
        "data": [
            {
                "image": base64.encodebytes(read_image(base_image1)).decode("utf-8"),
                "mask_image": base64.encodebytes(read_image(mask_image1)).decode(
                    "utf-8"
                ),
                "prompt": "Pikachu, cinematic, digital art, sitting on bench",
            },
            {
                "image": base64.encodebytes(read_image(base_image2)).decode("utf-8"),
                "mask_image": base64.encodebytes(read_image(mask_image2)).decode(
                    "utf-8"
                ),
                "prompt": "A woman with red hair in the style of Tamara de Lempicka.",
            },
        ],
    }
    pd.DataFrame(**input_data).to_csv(
        os.path.join(payload_path, "input2.csv"), index=False
    )


def prepare_online_payload(payload_path: str) -> None:
    """Prepare payload for online deployment.

    :param payload_path: Path to payload json file.
    :type payload_path: str
    :return: None
    """
    base_directory = os.path.dirname(os.path.dirname(__file__))

    base_image = os.path.join(
        base_directory, "inpainting_data", "images", "dog_on_bench.png"
    )
    mask_image = os.path.join(
        base_directory, "inpainting_data", "images", "dog_on_bench_mask.png"
    )

    request_json = {
        "input_data": {
            "columns": ["image", "mask_image", "prompt"],
            "index": [0],
            "data": [
                {
                    "image": base64.encodebytes(read_image(base_image)).decode("utf-8"),
                    "mask_image": base64.encodebytes(read_image(mask_image)).decode(
                        "utf-8"
                    ),
                    "prompt": "A yellow cat, high resolution, sitting on a park bench",
                }
            ],
        }
    }

    payload_path = os.path.join(base_directory, payload_path)
    with open(payload_path, "w") as request_file:
        json.dump(request_json, request_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare sample data for inpainting")
    parser.add_argument("--payload-path", type=str, help="payload file/ folder path")
    parser.add_argument(
        "--mode",
        type=str,
        default="online",
        help="Generate payload for online or batch deployment.",
    )
    args, unknown = parser.parse_known_args()

    if args.mode == "online":
        prepare_online_payload(args.payload_path)
    else:
        prepare_batch_payload(args.payload_path)
