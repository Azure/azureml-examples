# Test remote Batch AI

# Copyright (c) Microsoft Corporation. All rights reserved.
#
# Licensed under the MIT License.

from download_run_files import download_run_files

# Download files for the remote run.
download_run_files(
    experiment_names=["automl-regression-hardware-explain"], download_all_runs=True
)
