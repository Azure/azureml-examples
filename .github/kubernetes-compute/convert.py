import argparse
import yaml
import os
import json

from azure.ai.ml import MLClient
from azureml.core.authentication import AzureCliAuthentication


def convert(input_file, compute_target, instance_type, common_runtime, output_file):
    def _convert(input_file, data, job_schema):
        # check job type
        is_pipeline_job = False
        is_sweep_job = False
        if "pipelineJob" in job_schema or "jobs" in data:
            is_pipeline_job = True
        if "sweepJob" in job_schema or data.get("type") == "sweep":
            is_sweep_job = True

        print("Job type: pipelineJob", is_pipeline_job, "sweepJob:", is_sweep_job)

        # change compute target
        if compute_target:
            data["compute"] = "azureml:%s" % compute_target
            if is_pipeline_job:
                settings = data.get("settings", {})
                settings["default_compute"] = "azureml:%s" % compute_target
                data["settings"] = settings

        # set instance type
        if not is_pipeline_job and instance_type:
            resources = data.get("resources", {})
            resources["instance_type"] = instance_type
            data["resources"] = resources

        for field in ["trial", "component"]:
            if field not in data:
                continue

            file_field = data[field]
            if not isinstance(file_field, str):
                continue

            if file_field.startswith("file:"):
                file_field = file_field.split(":", 1)[1]

            print("Found sub job spec:", file_field)
            dirname = os.path.dirname(input_file)
            convert(
                os.path.join(dirname, file_field),
                compute_target,
                instance_type,
                common_runtime,
                "",
            )

        if is_pipeline_job:
            jobs = data.get("jobs", {})
            for step in jobs:
                print("Found step:", step)
                _convert(input_file, jobs[step], "")
            return

    print("Processing file:", input_file)
    if not os.path.exists(input_file):
        print("Warning: File doesn't exist: ", input_file)
        return
    with open(input_file, "r") as f:
        data = yaml.load(f, Loader=yaml.FullLoader)
        job_schema = data.get("$schema", "")
        _convert(input_file, data, job_schema)

        # write to output file if output file is specified, otherwise change inplace.
        if output_file:
            with open(output_file, "w") as f:
                yaml.dump(data, f)
        else:
            with open(input_file, "w") as f:
                yaml.dump(data, f)


def create_jsonl_files(job_dir, uri_folder_data_path):
    print("Creating jsonl files")
    src_images = "./data/fridgeObjects/"

    # We'll copy each JSONL file within its related MLTable folder
    training_mltable_path = os.path.join( root_dir, "./data/training-mltable-folder/")
    validation_mltable_path = os.path.join( root_dir, "./data/validation-mltable-folder/")

    train_validation_ratio = 5

    # Path to the training and validation files
    train_annotations_file = os.path.join(
        training_mltable_path, "train_annotations.jsonl"
    )
    validation_annotations_file = os.path.join(
        validation_mltable_path, "validation_annotations.jsonl"
    )

    # Baseline of json line dictionary
    json_line_sample = {"image_url": uri_folder_data_path, "label": ""}

    index = 0
    # Scan each sub directary and generate a jsonl line per image, distributed on train and valid JSONL files
    with open(train_annotations_file, "w") as train_f:
        with open(validation_annotations_file, "w") as validation_f:
            for className in os.listdir(src_images):
                subDir = src_images + className
                if not os.path.isdir(subDir):
                    continue
                # Scan each sub directary
                print("Parsing " + subDir)
                for image in os.listdir(subDir):
                    json_line = dict(json_line_sample)
                    json_line["image_url"] += f"{className}/{image}"
                    json_line["label"] = className

                    if index % train_validation_ratio == 0:
                        # validation annotation
                        validation_f.write(json.dumps(json_line) + "\n")
                    else:
                        # train annotation
                        train_f.write(json.dumps(json_line) + "\n")
                    index += 1
    print("done")


def prepare_data_for_automl_image_jobs(subscription, resource_group, workspace, data_asset_name ,job_dir):
    
    subscription_id = subscription
    resource_group = resource_group
    workspace = workspace
    
    credential = AzureCliAuthentication()
    ml_client = MLClient(credential, subscription_id, resource_group, workspace)
        
    latest_version = "1"
    for i in ml_client.data.list():
        if i.name == data_asset_name:
            version = i.latest_version
            break
    uri_folder_data_asset = ml_client.data.get(data_asset_name, latest_version)

    print(uri_folder_data_asset)
    print("")
    print("Path to folder in Blob Storage:")
    print(uri_folder_data_asset.path)
    create_jsonl_files(uri_folder_data_asset.path, job_dir)
    

if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Convert test case to AMLARC-compatible files."
    )
    parser.add_argument("-a", "--action", required=False, help="Action to take")
    parser.add_argument("-sub", "--subscription", required=False, help="subscription")
    parser.add_argument("-rg", "--resource-group", required=False, help="resource group")
    parser.add_argument("-ws", "--wrokspace", required=False, help="wrokspace")
    parser.add_argument("-dn", "--data-asset-name", required=False, help="data asset name")
    parser.add_argument("-jd", "--job-dir", required=False, help="dir of job")
    parser.add_argument("-i", "--input", required=False, help="Input test case file")
    parser.add_argument(
        "-o",
        "--output",
        required=False,
        help="Output AMLARC-compatible file, if not provides, " "replace file inplace",
    )
    parser.add_argument("-c", "--compute-target", required=False, help="Compute target")
    parser.add_argument("-it", "--instance-type", required=False, help="Instance type")
    parser.add_argument(
        "-cr",
        "--common-runtime",
        required=False,
        default=False,
        action="store_true",
        help='Enable common runtime explicitly, default is "false"',
    )
    args = parser.parse_args()
    
    if args.action == "prepare_data_for_automl_image_jobs":
        prepare_data_for_automl_image_jobs(
            arg.subscription,
            arg.resource_group, 
            arg.workspace, 
            arg.data_asset_name, 
            arg.job_dir
        )
    else:
        convert(
            args.input,
            args.compute_target,
            args.instance_type,
            args.common_runtime,
            args.output,
        )
