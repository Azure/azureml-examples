# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------

import argparse
import json
import logging
import os
import shutil
from typing import Any

from azureml.core import Run

from responsibleai import RAIInsights, __version__ as responsibleai_version

from constants import DashboardInfo, PropertyKeyValues
from arg_helpers import get_from_args, json_empty_is_none_parser
from rai_component_utilities import load_dataset, fetch_model_id, load_mlflow_model

_logger = logging.getLogger(__file__)
logging.basicConfig(level=logging.INFO)


def parse_args():
    # setup arg parser
    parser = argparse.ArgumentParser()

    parser.add_argument("--title", type=str, required=True)

    parser.add_argument(
        "--task_type", type=str, required=True, choices=["classification", "regression"]
    )

    parser.add_argument(
        "--model_info_path", type=str, help="name:version", required=True
    )

    parser.add_argument("--train_dataset", type=str, required=True)
    parser.add_argument("--test_dataset", type=str, required=True)

    parser.add_argument("--target_column_name", type=str, required=True)

    parser.add_argument("--maximum_rows_for_test_dataset", type=int, default=5000)
    parser.add_argument(
        "--categorical_column_names", type=str, help="Optional[List[str]]"
    )

    parser.add_argument("--classes", type=str, help="Optional[List[str]]")

    parser.add_argument("--output_path", type=str, help="Path to output JSON")

    # parse args
    args = parser.parse_args()

    # return args
    return args


def create_constructor_arg_dict(args):
    """Create a kwarg dict for RAIInsights constructor

    Only does the 'parameters' for the component, not the
    input ports
    """
    result = dict()

    cat_col_names = get_from_args(
        args, "categorical_column_names", custom_parser=json.loads, allow_none=True
    )
    class_names = get_from_args(
        args, "classes", custom_parser=json_empty_is_none_parser, allow_none=True
    )

    result["target_column"] = args.target_column_name
    result["task_type"] = args.task_type
    result["categorical_features"] = cat_col_names
    result["classes"] = class_names
    result["maximum_rows_for_test"] = args.maximum_rows_for_test_dataset

    return result


def copy_input_data(component_input_path: str, output_path: str):
    if os.path.isdir(component_input_path):
        src_path = component_input_path
    else:
        src_path = os.path.dirname(component_input_path)
    src_path = src_path + "/"
    _logger.info(f"Copying from {src_path} to {output_path}")
    assert os.path.isdir(src_path), "Checking src_path"
    shutil.copytree(src=src_path, dst=output_path)


def main(args):

    my_run = Run.get_context()

    _logger.info("Dealing with initialization dataset")
    train_df = load_dataset(args.train_dataset)

    _logger.info("Dealing with evaluation dataset")
    test_df = load_dataset(args.test_dataset)

    model_id = fetch_model_id(args.model_info_path)
    _logger.info("Loading model: {0}".format(model_id))
    model_estimator = load_mlflow_model(my_run.experiment.workspace, model_id)

    constructor_args = create_constructor_arg_dict(args)

    # Make sure that it actually loads
    _logger.info("Creating RAIInsights object")
    _ = RAIInsights(
        model=model_estimator, train=train_df, test=test_df, **constructor_args
    )

    _logger.info("Saving JSON for tool components")
    output_dict = {
        DashboardInfo.RAI_INSIGHTS_RUN_ID_KEY: str(my_run.id),
        DashboardInfo.RAI_INSIGHTS_MODEL_ID_KEY: model_id,
        DashboardInfo.RAI_INSIGHTS_CONSTRUCTOR_ARGS_KEY: constructor_args,
    }
    output_file = os.path.join(
        args.output_path, DashboardInfo.RAI_INSIGHTS_PARENT_FILENAME
    )
    with open(output_file, "w") as of:
        json.dump(output_dict, of)

    _logger.info("Copying train data files")
    copy_input_data(
        args.train_dataset,
        os.path.join(args.output_path, DashboardInfo.TRAIN_FILES_DIR),
    )
    _logger.info("Copying test data files")
    copy_input_data(
        args.test_dataset, os.path.join(args.output_path, DashboardInfo.TEST_FILES_DIR)
    )


# run script
if __name__ == "__main__":
    # add space in logs
    print("*" * 60)
    print("\n\n")

    # parse args
    args = parse_args()

    # run main function
    main(args)

    # add space in logs
    print("*" * 60)
    print("\n\n")
