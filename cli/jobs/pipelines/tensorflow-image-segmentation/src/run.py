# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""
This script implements a Distributed Tensorflow training sequence.

IMPORTANT: We have tagged the code with the following expressions to walk you through
the key implementation details.

Using your editor, search for those strings to get an idea of how to implement:
- DISTRIBUTED : how to implement distributed tensorflow,
- MLFLOW : how to implement mlflow reporting of metrics and artifacts,
- PROFILER: how to implement tensorflow profiling within a job.
"""
import os
import sys
import time
import logging
import argparse
from distutils.util import strtobool

import mlflow

# tensorflow imports
# import tensorflow as tf
# from tensorflow import keras


# internal imports
## non-specific helper code
# from common.profiling import LogTimeBlock, LogDiskIOBlock  # noqa : E402

## tensorflow generic helping code
# from tf_helper.training import TensorflowDistributedModelTrainingSequence  # noqa : E402

## classification specific code
# from segmentation.model import load_model  # noqa : E402
# from segmentation.io import ImageAndMaskSequenceDataset  # noqa : E402


SCRIPT_START_TIME = time.time()  # just to measure time to start


def run(args):
    """Run the script using CLI arguments"""
    logger = logging.getLogger(__name__)
    logger.info(f"Running with arguments: {args}")
    print("args : {}".format(args))

    # MLFLOW: initialize mlflow (once in entire script)
    mlflow.start_run()

    from mlflow.tracking import MlflowClient
    client = MlflowClient()
    run = mlflow.active_run()
    run = client.get_run(run.info.run_id)

    print("location : mlflow.start_run")
    print("run_id : {}".format(run.info.run_id))
    print("run_params : {}".format(run.data.params))
  



def main(cli_args=None):
    """Main function of the script."""
    args = None
    # run the run function
    run(args)


if __name__ == "__main__":
    main()
