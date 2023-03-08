import os
from pathlib import Path
from random import randint
from uuid import uuid4

from azure.ml.component import dsl, Environment
from azure.ml.component.dsl.types import Input, Output, Integer
from azureml.core import Dataset, Datastore, Run

# init customer environment with conda YAML
# the YAML file shall be put under your code folder.
conda_env = Environment(conda_file=Path(__file__).parent / "conda.yaml")

version = "1"

# init customer environment with environment YAML
# env = Environment(conda_file = Path(__file__).parent / 'env.yaml')


@dsl.command_component(
    name="create_multi_path_dataset",
    display_name="create_multi_path_dataset",
    version=version,
    environment=conda_env,
)
def create_multi_path_dataset(
    data_output: Output,
):
    run = Run.get_context()
    ws = run.experiment.workspace
    datastore = Datastore.get(ws, "workspaceblobstore")
    relative_path_1 = "LocalUpload/4c8f25cc30097999fb052b06bca2a561/data/"
    relative_path_2 = "LocalUpload/cc355250825d6284521e9ae14f3db123/src/"
    relative_path = f'"{relative_path_1}", "{relative_path_2}"'

    # Create dataset object
    file_dataset = Dataset.File.from_files(path=[(datastore, relative_path)], validate=False)
    file_dataset._ensure_saved_internal(ws)

    run.output_datasets["data_output"].link(file_dataset)


@dsl.command_component(
    name="consume_dataset",
    display_name="consume_dataset",
    version=version,
)
def consume_dataset(
    data_path: Input,
):
    """Consume a dataset input"""
    data = Path(data_path)
    is_folder = data.is_dir()
    print(f"Data path is a folder: {is_folder}")
    print(f"Data path: {data}")
    if is_folder:
        print("List all contents in data folder:")
        list_files(str(data))
    else:
        print(f"Data file name: {data.name}")
        print(f"Data file content: \n\t{data.read_text()}")


def list_files(startpath):
    for root, dirs, files in os.walk(startpath):
        level = root.replace(startpath, '').count(os.sep)
        indent = ' ' * 4 * (level)
        print('{}{}/'.format(indent, os.path.basename(root)))
        subindent = ' ' * 4 * (level + 1)
        for f in files:
            print('{}{}'.format(subindent, f))