"""download_models

Downloads models needed for Triton example notebooks.
"""
import os
import urllib
from azure.storage.blob import BlobClient


model_names = ["densenet_onnx", "bidaf-9"]


def download_triton_models(prefix):
    for model in model_names:
        folder_path, model_file_path = _generate_paths(model, prefix)
        url = f"https://aka.ms/{model}-model"
        _download_model(model_file_path, folder_path, url)
        print(f"successfully downloaded model: {model}")


def delete_triton_models(prefix):
    for model in model_names:
        _, model_file_path = _generate_paths(model, prefix)
        try:
            os.remove(model_file_path)
            print(f"successfully deleted model: {model}")
        except FileNotFoundError:
            print(f"model: {model} was already deleted")


def _download_model(model_file_path, folder_path, url):
    response = urllib.request.urlopen(url)

    blob_client = BlobClient.from_blob_url(response.url)

    # save the model if it does not already exist
    if not os.path.exists(model_file_path):
        os.makedirs(folder_path, exist_ok=True)
        with open(model_file_path, "wb") as my_blob:
            download_stream = blob_client.download_blob()
            my_blob.write(download_stream.readall())


def _generate_paths(model, prefix):
    folder_path = prefix.joinpath("models", "triton", model, "1")
    model_file_path = prefix.joinpath(folder_path, "model.onnx")
    return folder_path, model_file_path
