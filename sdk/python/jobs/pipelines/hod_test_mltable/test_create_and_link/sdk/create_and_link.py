from pathlib import Path

from mldesigner import command_component, Input, Output
# from mldesigner.dsl._dynamic_executor import DynamicExecutor
from azure.ai.ml.entities import Data


@command_component(environment="./conda.yaml")
def get_my_data() -> Output(type="string", is_control=True):
    """Get a data asset and pass to next component"""
    # ml_client = DynamicExecutor._get_ml_client()
    # my_data = ml_client.data.get(name="mltable_test_files", version="2")
    # print(f"Data id: {my_data.id!r}")

    asset_id = ""
    return asset_id


@command_component(environment="./conda.yaml")
def consume_data(data_path: Input(type="uri_folder")):
    """Consume an input"""
    data = Path(data_path)
    is_file = data.is_file()
    print(f"Data path is a file: {data.is_dir()}")
    if is_file:
        print("List all contents in data folder:")
        for item in data.glob("**/*"):
            file_type = "File" if item.is_file() else "Folder"
            print(f"\t{file_type}: {item.name}")
    else:
        print(f"Data file name: {data.name}")