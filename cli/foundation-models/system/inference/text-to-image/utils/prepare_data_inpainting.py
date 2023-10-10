# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

# Prepare request payload for the inpainting task

import argparse
import glob
import json
import io
import base64
import os
import pandas as pd
from PIL import Image
from pathlib import Path


def read_image(image_path: str) -> bytes:
    """Reads an image from a file path into a byte array.

    :param image_path: Path to image file.
    :type image_path: str
    :return: Byte array of image.
    :rtype: bytes
    """
    with open(image_path, "rb") as f:
        return f.read()


def prepare_batch_payload(payload_path: str) -> None:
    """Prepare payload for online deployment.

    :param payload_path: Path to payload csv file.
    :type payload_path: str
    :return: None
    """

    base_image1 = "inpainting_data/images/dog_on_bench.png"
    mask_image1 = "inpainting_data/masks/dog_on_bench.png"
    base_image2 = "inpainting_data/images/teapot.png"
    mask_image2 = "inpainting_data/masks/teapot.png"

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

    # Use glob to get a list of CSV files in the folder
    csv_files = glob.glob(os.path.join(payload_path, "*.csv"))

    # Read all CSV files into a single DataFrame using pd.concat
    batch_df = pd.concat((pd.read_csv(file) for file in csv_files), ignore_index=True)

    # Specify the folder where your CSV files should be saved
    processed_dataset_parent_dir = os.path.join(payload_path, "processed_batch_data")
    os.makedirs(processed_dataset_parent_dir, exist_ok=True)
    batch_input_file = "batch_input.csv"

    # Divide this into files of <x> rows each
    batch_size_per_predict = 2
    for i in range(0, len(batch_df), batch_size_per_predict):
        j = i + batch_size_per_predict
        batch_df[i:j].to_csv(
            os.path.join(processed_dataset_parent_dir, str(i) + batch_input_file)
        )

    # Check out the first and last file name created
    input_paths = sorted(
        Path(processed_dataset_parent_dir).iterdir(), key=os.path.getmtime
    )
    input_files = [os.path.basename(path) for path in input_paths]
    print(f"{input_files[0]} to {str(i)}{batch_input_file}.")


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
        base_directory, "inpainting_data", "masks", "dog_on_bench.png"
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
    print(base_directory)
    print(payload_path)
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
