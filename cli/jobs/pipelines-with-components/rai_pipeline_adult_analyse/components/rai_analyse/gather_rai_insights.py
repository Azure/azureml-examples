# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------

import argparse
import json
import logging
from pathlib import Path
import shutil
import tempfile

from typing import Dict

from azureml.core import Run

from responsibleai import RAIInsights
from responsibleai.serialization_utilities import serialize_json_safe

from constants import DashboardInfo, RAIToolType
from rai_component_utilities import (
    create_rai_tool_directories,
    create_rai_insights_from_port_path,
    copy_insight_to_raiinsights,
    load_dashboard_info_file,
    add_properties_to_gather_run,
    print_dir_tree,
)

_DASHBOARD_CONSTRUCTOR_MISMATCH = (
    "Insight {0} was not " "computed from the constructor specified"
)
_DUPLICATE_TOOL = "Insight {0} is of type {1} which is already present"

_logger = logging.getLogger(__file__)
logging.basicConfig(level=logging.INFO)


def parse_args():
    # setup arg parser
    parser = argparse.ArgumentParser()

    parser.add_argument("--constructor", type=str, required=True)
    parser.add_argument("--insight_1", type=str, default=None)
    parser.add_argument("--insight_2", type=str, default=None)
    parser.add_argument("--insight_3", type=str, default=None)
    parser.add_argument("--insight_4", type=str, default=None)
    parser.add_argument("--dashboard", type=str, required=True)
    parser.add_argument("--ux_json", type=str, required=True)

    # parse args
    args = parser.parse_args()

    # return args
    return args


def main(args):
    dashboard_info = load_dashboard_info_file(args.constructor)
    _logger.info("Constructor info: {0}".format(dashboard_info))

    with tempfile.TemporaryDirectory() as incoming_temp_dir:
        incoming_dir = Path(incoming_temp_dir)

        my_run = Run.get_context()
        rai_temp = create_rai_insights_from_port_path(my_run, args.constructor)
        rai_temp.save(incoming_temp_dir)

        print("Savied rai_temp")
        print_dir_tree(incoming_temp_dir)
        print("=======")

        create_rai_tool_directories(incoming_dir)
        _logger.info("Saved empty RAI Insights input to temporary directory")

        insight_paths = [args.insight_1, args.insight_2, args.insight_3, args.insight_4]

        included_tools: Dict[str, bool] = {
            RAIToolType.CAUSAL: False,
            RAIToolType.COUNTERFACTUAL: False,
            RAIToolType.ERROR_ANALYSIS: False,
            RAIToolType.EXPLANATION: False,
        }
        for i in range(len(insight_paths)):
            current_insight_arg = insight_paths[i]
            if current_insight_arg is not None:
                current_insight_path = Path(current_insight_arg)
                _logger.info("Checking dashboard info")
                insight_info = load_dashboard_info_file(current_insight_path)
                _logger.info("Insight info: {0}".format(insight_info))

                # Cross check against the constructor
                if insight_info != dashboard_info:
                    err_string = _DASHBOARD_CONSTRUCTOR_MISMATCH.format(i + 1)
                    raise ValueError(err_string)

                # Copy the data
                _logger.info("Copying insight {0}".format(i + 1))
                tool = copy_insight_to_raiinsights(incoming_dir, current_insight_path)

                # Can only have one instance of each tool
                if included_tools[tool]:
                    err_string = _DUPLICATE_TOOL.format(i + 1, tool)
                    raise ValueError(err_string)

                included_tools[tool] = True
            else:
                _logger.info("Insight {0} is None".format(i + 1))

        _logger.info("Tool summary: {0}".format(included_tools))

        rai_i = RAIInsights.load(incoming_dir)
        _logger.info("Object loaded")

        rai_i.save(args.dashboard)
        _logger.info("Saved dashboard to oputput")

        rai_data = rai_i.get_data()
        rai_dict = serialize_json_safe(rai_data)
        json_filename = "dashboard.json"
        output_path = Path(args.ux_json) / json_filename
        with open(output_path, "w") as json_file:
            json.dump(rai_dict, json_file)
        _logger.info("Dashboard JSON written")

        add_properties_to_gather_run(dashboard_info, included_tools)
        _logger.info("Processing completed")


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
