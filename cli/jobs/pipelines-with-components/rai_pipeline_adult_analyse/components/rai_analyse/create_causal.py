# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------

import argparse
import json
import logging

from pathlib import Path
from shutil import copyfile

from responsibleai import RAIInsights

from azureml.core import Run

from constants import RAIToolType, DashboardInfo
from rai_component_utilities import (
    create_rai_insights_from_port_path,
    save_to_output_port,
    copy_dashboard_info_file,
)
from arg_helpers import (
    float_or_json_parser,
    boolean_parser,
    str_or_list_parser,
    int_or_none_parser,
)

_logger = logging.getLogger(__file__)
logging.basicConfig(level=logging.INFO)


def parse_args():
    # setup arg parser
    parser = argparse.ArgumentParser()

    parser.add_argument("--rai_insights_dashboard", type=str, required=True)

    parser.add_argument("--treatment_features", type=json.loads, help="List[str]")
    parser.add_argument(
        "--heterogeneity_features",
        type=json.loads,
        help="Optional[List[str]] use 'null' to skip",
    )
    parser.add_argument("--nuisance_model", type=str)
    parser.add_argument("--heterogeneity_model", type=str)
    parser.add_argument("--alpha", type=float)
    parser.add_argument("--upper_bound_on_cat_expansion", type=int)
    parser.add_argument(
        "--treatment_cost",
        type=float_or_json_parser,
        help="Union[float, List[Union[float, np.ndarray]]]",
    )
    parser.add_argument("--min_tree_leaf_samples", type=int)
    parser.add_argument("--max_tree_depth", type=int)
    parser.add_argument("--skip_cat_limit_checks", type=boolean_parser)
    parser.add_argument("--categories", type=str_or_list_parser)
    parser.add_argument("--n_jobs", type=int)
    parser.add_argument("--verbose", type=int)
    parser.add_argument("--random_state", type=int_or_none_parser)

    parser.add_argument("--causal_path", type=str)

    # parse args
    args = parser.parse_args()

    # return args
    return args


def main(args):
    my_run = Run.get_context()
    # Load the RAI Insights object
    rai_i: RAIInsights = create_rai_insights_from_port_path(
        my_run, args.rai_insights_dashboard
    )

    # Add the causal analysis
    rai_i.causal.add(
        treatment_features=args.treatment_features,
        heterogeneity_features=args.heterogeneity_features,
        nuisance_model=args.nuisance_model,
        heterogeneity_model=args.heterogeneity_model,
        alpha=args.alpha,
        upper_bound_on_cat_expansion=args.upper_bound_on_cat_expansion,
        treatment_cost=args.treatment_cost,
        min_tree_leaf_samples=args.min_tree_leaf_samples,
        max_tree_depth=args.max_tree_depth,
        skip_cat_limit_checks=args.skip_cat_limit_checks,
        categories=args.categories,
        n_jobs=args.n_jobs,
        verbose=args.verbose,
        random_state=args.random_state,
    )
    _logger.info("Added causal")

    # Compute
    rai_i.compute()
    _logger.info("Computation complete")

    # Save
    save_to_output_port(rai_i, args.causal_path, RAIToolType.CAUSAL)
    _logger.info("Saved computation to output port")

    # Copy the dashboard info file
    copy_dashboard_info_file(args.rai_insights_dashboard, args.causal_path)

    _logger.info("Completing")


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
