# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------

import json
import logging
import os
import pathlib
import shutil
import tempfile
import uuid

from typing import Any, Dict
from zipfile import Path

import pandas as pd

import mlflow

from azureml.core import Model, Run, Workspace

from responsibleai import RAIInsights, __version__ as responsibleai_version

from constants import DashboardInfo, PropertyKeyValues, RAIToolType


_logger = logging.getLogger(__file__)
logging.basicConfig(level=logging.INFO)


# Directory names saved by RAIInsights might not match tool names
_tool_directory_mapping: Dict[str, str] = {
    RAIToolType.CAUSAL: "causal",
    RAIToolType.COUNTERFACTUAL: "counterfactual",
    RAIToolType.ERROR_ANALYSIS: "error_analysis",
    RAIToolType.EXPLANATION: "explainer",
}


def print_dir_tree(base_dir):
    print("\nBEGIN DIRTREE")
    for current_dir, subdirs, files in os.walk(base_dir):
        # Current Iteration Directory
        print(current_dir)

        # Directories
        for dirname in sorted(subdirs):
            print("\t" + dirname + "/")

        # Files
        for filename in sorted(files):
            print("\t" + filename)
    print("END DIRTREE\n", flush=True)


def fetch_model_id(model_info_path: str):
    model_info_path = os.path.join(model_info_path, DashboardInfo.MODEL_INFO_FILENAME)
    with open(model_info_path, "r") as json_file:
        model_info = json.load(json_file)
    return model_info[DashboardInfo.MODEL_ID_KEY]


def load_mlflow_model(workspace: Workspace, model_id: str) -> Any:
    mlflow.set_tracking_uri(workspace.get_mlflow_tracking_uri())

    model = Model._get(workspace, id=model_id)
    model_uri = "models:/{}/{}".format(model.name, model.version)
    return mlflow.pyfunc.load_model(model_uri)._model_impl


def load_dataset(parquet_path: str):
    _logger.info("Loading parquet file: {0}".format(parquet_path))
    df = pd.read_parquet(parquet_path)
    print(df.dtypes)
    print(df.head(10))
    return df


def load_dashboard_info_file(input_port_path: str) -> Dict[str, str]:
    # Load the rai_insights_dashboard file info
    rai_insights_dashboard_file = os.path.join(
        input_port_path, DashboardInfo.RAI_INSIGHTS_PARENT_FILENAME
    )
    with open(rai_insights_dashboard_file, "r") as si:
        dashboard_info = json.load(si)
    _logger.info("rai_insights_parent info: {0}".format(dashboard_info))
    return dashboard_info


def copy_dashboard_info_file(src_port_path: str, dst_port_path: str):
    src = pathlib.Path(src_port_path) / DashboardInfo.RAI_INSIGHTS_PARENT_FILENAME
    dst = pathlib.Path(dst_port_path) / DashboardInfo.RAI_INSIGHTS_PARENT_FILENAME

    shutil.copyfile(src, dst)


def create_rai_tool_directories(rai_insights_dir: pathlib.Path) -> None:
    # Have to create empty subdirectories for the managers
    # THe RAI Insights object expect these to be present, but
    # since directories don't actually exist in Azure Blob store
    # they may not be present (some of the tools always have
    # a file present, even if no tool instances have been added)
    for v in _tool_directory_mapping.values():
        os.makedirs(rai_insights_dir / v, exist_ok=True)
    _logger.info("Added empty directories")


def load_rai_insights_from_input_port(input_port_path: str) -> RAIInsights:
    with tempfile.TemporaryDirectory() as incoming_temp_dir:
        incoming_dir = pathlib.Path(incoming_temp_dir)
        shutil.copytree(input_port_path, incoming_dir, dirs_exist_ok=True)
        _logger.info("Copied RAI Insights input to temporary directory")

        create_rai_tool_directories(incoming_dir)

        result = RAIInsights.load(incoming_dir)
        _logger.info("Loaded RAIInsights object")
    return result


def copy_insight_to_raiinsights(
    rai_insights_dir: pathlib.Path, insight_dir: pathlib.Path
) -> str:
    print("Starting copy")

    # Recall that we copy the JSON containing metadata from the
    # constructor component into each directory
    # This means we have that file and the results directory
    # present in the insight_dir
    dir_items = list(insight_dir.iterdir())
    assert len(dir_items) == 2

    # We want the directory, not the JSON file
    if dir_items[0].name == DashboardInfo.RAI_INSIGHTS_PARENT_FILENAME:
        tool_dir_name = dir_items[1].name
    else:
        tool_dir_name = dir_items[0].name

    _logger.info("Detected tool: {0}".format(tool_dir_name))
    assert tool_dir_name in _tool_directory_mapping.values()
    for k, v in _tool_directory_mapping.items():
        if tool_dir_name == v:
            tool_type = k
    _logger.info("Mapped to tool: {0}".format(tool_type))
    tool_dir = insight_dir / tool_dir_name

    tool_dir_items = list(tool_dir.iterdir())
    assert len(tool_dir_items) == 1

    if tool_type == RAIToolType.EXPLANATION:
        # Explanations will have a directory already present for some reason
        # Furthermore we only support one explanation per dashboard for
        # some other reason
        # Put together, if we have an explanation, we need to remove
        # what's there already or we can get confused
        _logger.info("Detected explanation, removing existing directory")
        for item in (rai_insights_dir / tool_dir_name).iterdir():
            _logger.info("Removing directory {0}".format(str(item)))
            shutil.rmtree(item)

    src_dir = insight_dir / tool_dir_name / tool_dir_items[0].parts[-1]
    dst_dir = rai_insights_dir / tool_dir_name / tool_dir_items[0].parts[-1]
    shutil.copytree(src=src_dir, dst=dst_dir)

    _logger.info("Copy complete")
    return tool_type


def save_to_output_port(rai_i: RAIInsights, output_port_path: str, tool_type: str):
    with tempfile.TemporaryDirectory() as tmpdirname:
        rai_i.save(tmpdirname)
        _logger.info(f"Saved to {tmpdirname}")

        tool_dir_name = _tool_directory_mapping[tool_type]
        insight_dirs = os.listdir(pathlib.Path(tmpdirname) / tool_dir_name)
        assert len(insight_dirs) == 1, "Checking for exactly one tool output"
        _logger.info("Checking dirname is GUID")
        uuid.UUID(insight_dirs[0])

        target_path = pathlib.Path(output_port_path) / tool_dir_name
        target_path.mkdir()
        _logger.info("Created output directory")

        _logger.info("Starting copy")
        shutil.copytree(
            pathlib.Path(tmpdirname) / tool_dir_name, target_path, dirs_exist_ok=True
        )
    _logger.info("Copied to output")


def add_properties_to_gather_run(
    dashboard_info: Dict[str, str], tool_present_dict: Dict[str, str]
):
    _logger.info("Adding properties to the gather run")
    gather_run = Run.get_context()

    run_properties = {
        PropertyKeyValues.RAI_INSIGHTS_TYPE_KEY: PropertyKeyValues.RAI_INSIGHTS_TYPE_GATHER,
        PropertyKeyValues.RAI_INSIGHTS_RESPONSIBLEAI_VERSION_KEY: responsibleai_version,
        PropertyKeyValues.RAI_INSIGHTS_MODEL_ID_KEY: dashboard_info[
            DashboardInfo.RAI_INSIGHTS_MODEL_ID_KEY
        ],
    }

    _logger.info("Appending tool present information")
    for k, v in tool_present_dict.items():
        key = PropertyKeyValues.RAI_INSIGHTS_TOOL_KEY_FORMAT.format(k)
        run_properties[key] = str(v)

    _logger.info("Making service call")
    gather_run.add_properties(run_properties)
    _logger.info("Properties added to gather run")


def create_rai_insights_from_port_path(my_run: Run, port_path: str) -> RAIInsights:
    _logger.info("Creating RAIInsights from constructor component output")

    _logger.info("Loading data files")
    df_train = load_dataset(os.path.join(port_path, DashboardInfo.TRAIN_FILES_DIR))
    df_test = load_dataset(os.path.join(port_path, DashboardInfo.TEST_FILES_DIR))

    _logger.info("Loading config file")
    config = load_dashboard_info_file(port_path)
    constructor_args = config[DashboardInfo.RAI_INSIGHTS_CONSTRUCTOR_ARGS_KEY]

    _logger.info("Loading model")
    model_id = config[DashboardInfo.RAI_INSIGHTS_MODEL_ID_KEY]
    _logger.info("Loading model: {0}".format(model_id))
    model_estimator = load_mlflow_model(my_run.experiment.workspace, model_id)

    _logger.info("Creating RAIInsights object")
    rai_i = RAIInsights(
        model=model_estimator, train=df_train, test=df_test, **constructor_args
    )
    return rai_i
