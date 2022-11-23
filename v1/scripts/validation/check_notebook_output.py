# This is used in notebook validation to check the output cells of the notebook.
# It checks for unexpected warnings or errors
# The parameters are:
# 	--file_name   The name of the notebook output file
#       --folder      The path for the folder containing the notebook output.
#       --check       A list of strings to check for.
#                     stderr indicates anything written to stderr.

import json
import requests
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--file_name")
parser.add_argument("--folder")
parser.add_argument("--check", nargs="+")

inputArgs = parser.parse_args()
full_name = os.path.join(inputArgs.folder, inputArgs.file_name)

allowed_list = [
    "UserWarning: Matplotlib is building the font cache",
    "UserWarning: Starting from version 2.2.1",
    "the library file in distribution wheels for macOS is built by the Apple Clang",
    "brew install libomp",
    "No verified requirements file found" "failed to create lock file",
    "retrying",
    "Using default datastore for uploads",
    "Already registered authentication for run id",
    "INFO - Initializing logging file for interpret-community",
    "INFO:interpret_community.common.explanation_utils:Using default datastore for uploads",
    "better speed can be achieved with apex",
    "numExpr defaulting to",
    "no version information available",
    "Falling back to use azure cli login credentials",
    "recommend to use ServicePrincipalAuthentication or MsiAuthentication",
    "Please refer to aka.ms/aml-notebook-auth",
    "Class KubernetesCompute: This is an experimental class",
    "Class SynapseCompute: This is an experimental class",
    'Please use "Dataset.File.upload_directory"',
    'Please use "FileDatasetFactory.upload_directory" instead',
    "Called AzureBlobDatastore.upload_files",
    "LinkTabularOutputDatasetConfig",
    "This may take a few minutes",
    "Downloading dataset",
    "logger.warning",
    "Importing plotly failed",
]

with open(full_name, "r") as notebook_file:
    notebook = json.load(notebook_file)

    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            for output in cell["outputs"]:
                if "text" in output:
                    for line in output["text"]:
                        # Avoid failing notebook runs on empty
                        # warnings.
                        if not line.strip():
                            continue
                        for not_allowed in inputArgs.check:
                            lower_line = line.lower()
                            if not_allowed == "stderr":
                                if "name" in output:
                                    assert output["name"] != "stderr" or any(
                                        (a.lower() in lower_line) for a in allowed_list
                                    ), (
                                        "Found stderr line:\n"
                                        + line
                                        + "\n in file "
                                        + inputArgs.file_name
                                    )
                            else:
                                assert not_allowed.lower() not in lower_line or any(
                                    (a.lower() in lower_line) for a in allowed_list
                                ), (
                                    not_allowed
                                    + " found in line:\n"
                                    + line
                                    + "\n in file "
                                    + inputArgs.file_name
                                )

print("check notebook output completed")
