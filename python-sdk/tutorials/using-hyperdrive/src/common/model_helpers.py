"""
This is the file that contains Model Helpers for AzureML Pipelines
"""
import os
import json


def write_model_info(folder_path: str, filename: str, model_info: dict) -> str:
    """
    Write model information to given file path
    Parameters:
      folder_path: str: folder of file containing model information
      filename: str: filename of file contianing model information
      model_info: dict: model information to be written as JSON
    Returns:
      str: file path of newly created file containing model information
    """
    os.makedirs(folder_path, exist_ok=True)
    model_info_filepath = os.path.join(folder_path, filename)

    with open(model_info_filepath, "w") as output_file:
        json.dump(model_info, output_file)

    return model_info_filepath
