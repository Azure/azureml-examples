import json
import os
import urllib.request
import zipfile

from azureml.core import Dataset
from azureml.core.compute import AmlCompute, ComputeTarget
from azureml.core.compute_target import ComputeTargetException
from azureml.data import DataType


def create_compute_cluster(name, sku, workspace):
    """Create a compute cluster."""
    try:
        compute_target = ComputeTarget(workspace=workspace, name=name)
        print("Found existing cluster for {} -- using it".format(name))
        return compute_target
    except ComputeTargetException:
        compute_config = AmlCompute.provisioning_configuration(vm_size=sku, max_nodes=4, idle_seconds_before_scaledown=120)
        compute_target = ComputeTarget.create(workspace, name, compute_config)
        print("Creating cluster {}".format(name))
        compute_target.wait_for_completion(show_output=True)
        return compute_target


def download_and_unzip_file(url, destination_directory):
    """Download and unzip a remote url file to a destination on local disk."""
    file_name = url.split('/')[-1]
    zip_file_local_path = os.path.join(destination_directory, file_name)
    urllib.request.urlretrieve(url, zip_file_local_path)
    with zipfile.ZipFile(zip_file_local_path, 'r') as zip_ref:
        zip_ref.extractall(destination_directory)


def register_jsonl_dataset_from_coco_metadata(
        workspace,
        coco_metadata_file_path,
        output_jsonl_file_path,
        dataset_name, 
        dataset_path_in_datastore
    ):
    """Build and register a JSONL dataset from a MS COCO artifacts."""
    datastore = workspace.get_default_datastore()
    with open(coco_metadata_file_path, 'r') as f:
        instances = json.load(f)

    ppl_images = set()
    for annotation in instances['annotations']:
        if annotation['category_id'] == 1:
            ppl_images.add(annotation['image_id'])

    os.makedirs(os.path.dirname(output_jsonl_file_path), exist_ok=True)
            
    with open(output_jsonl_file_path, 'w') as f:
        for image in instances['images']:
            jsonl = json.dumps({
                'image_url': f"AmlDatastore://{datastore.name}/{dataset_path_in_datastore}/{image['file_name']}",
                'contains_person': True if image['id'] in ppl_images else False
            })
            f.write(f'{jsonl}\n')

    Dataset.File.upload_directory(
        src_dir=os.path.dirname(output_jsonl_file_path),
        target=(datastore, "coco"),
        show_progress=False,
        overwrite=True)
    dataset = Dataset.Tabular.from_json_lines_files(
        path=datastore.path(f"coco/{os.path.basename(output_jsonl_file_path)}"),
        set_column_types={"image_url": DataType.to_stream(workspace)},
    )
    dataset.register(workspace=workspace, name=dataset_name, create_new_version=True)


def get_az_storage_cli_auth_param(datastore):
    """Get auth param for az storage CLI command."""
    if datastore.account_key:
        return f'--account-key {datastore.account_key}'
    return f'--sas-token {datastore.sas_token}'
