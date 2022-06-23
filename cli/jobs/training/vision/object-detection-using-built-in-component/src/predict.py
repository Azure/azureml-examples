import argparse
import base64
import os
import urllib.request
from zipfile import ZipFile

import mlflow.pyfunc
import pandas as pd


if __name__ == "__main__":
    # Parse input arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str)
    args, _ = parser.parse_known_args()

    # Download images to score
    download_url = "https://cvbp-secondary.z19.web.core.windows.net/datasets/object_detection/odFridgeObjects.zip"
    data_file = "data/odFridgeObjects.zip"
    os.mkdir("data")
    print("Downloading images...")
    urllib.request.urlretrieve(download_url, filename=data_file)
    print("Download completed")

    # Extract downloaded images
    with ZipFile(data_file, "r") as zip:
        print("Extracting downloaded images...")
        zip.extractall(path="./data")

    # Images to score
    test_image_paths = [
        "./data/odFridgeObjects/images/1.jpg",
        "./data/odFridgeObjects/images/2.jpg",
        "./data/odFridgeObjects/images/3.jpg",
    ]

    # Define helper method to read the bytes of a file from disk
    def read_file_bytes(image_path):
        with open(image_path, "rb") as f:
            return f.read()

    # Read the test images into a pandas DataFrame
    test_df = pd.DataFrame(
        data=[
            base64.encodebytes(read_file_bytes(image_path))
            for image_path in test_image_paths
        ],
        columns=["image"],
    )
    print(f"Test image DataFrame shape: {test_df.shape}")

    # Load the trained model
    pyfunc_model = mlflow.pyfunc.load_model(args.model)

    # Score the test images
    result = pyfunc_model.predict(test_df).to_json(orient="records")

    # Print the predictions
    print("Predictions:")
    print(result)
