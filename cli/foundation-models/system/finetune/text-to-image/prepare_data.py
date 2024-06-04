import argparse
import json
import requests
import os
from typing import List, Dict


def get_data(url: str) -> List[Dict]:
    """Send a GET request to the specified URL, parse the JSON content of the response.

    :param url: The URL to send the GET request to.
    :type url: str
    :return: The "rows" field of the parsed data.
    :rtype: List[Dict]
    """
    response = requests.get(url)
    response.raise_for_status()
    data = json.loads(response.content)
    return data["rows"]


def download_images(data: List[Dict], dataset_dir: str) -> None:
    """Create a directory for the images and download each image to the directory.

    :param data: The parsed data.
    :type data: List[Dict]
    :param dataset_dir: The directory to save the images to.
    :type dataset_dir: str
    """
    # Create a directory for the images
    os.makedirs(dataset_dir, exist_ok=True)

    # Iterate over the parsed data and download each image
    for i, item in enumerate(data):
        image_url = item["row"]["image"]["src"]
        image_response = requests.get(image_url)

        # Check if the request was successful
        image_response.raise_for_status()

        # Write the image data to a file
        with open(os.path.join(dataset_dir, f"image_{i}.jpg"), "wb") as f:
            f.write(image_response.content)


if __name__ == "__main__":
    """
    Parse command-line arguments for the URL and directory name, and pass them to the
    get_data() and download_images() functions.
    """
    parser = argparse.ArgumentParser(description="Download images from a dataset.")
    parser.add_argument("--url", required=True, help="URL of the dataset.")
    parser.add_argument(
        "--dataset_dir", required=True, help="Directory to save the images."
    )
    args = parser.parse_args()

    data = get_data(args.url)
    download_images(data, args.dataset_dir)
