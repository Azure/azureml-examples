# This is used in notebook validation to check the output of individual cells of the notebook.
# The parameters are:
# 	--file_name              The name of the notebook output file
#       --folder                 The path for the folder containing the notebook output.
#       --expected_stdout        The expected output
#       --cell_source            Option cell source to be checked
#       --cell_output_substring  The specified cell is checked for this output.
#       --check_widget           True indicates that the widget output should be checked.

import json
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--file_name")
parser.add_argument("--folder")
parser.add_argument("--expected_stdout", nargs="*")
parser.add_argument("--cell_source", nargs="*")
parser.add_argument("--cell_output_substring")
parser.add_argument("--check_widget", type=bool)

inputArgs = parser.parse_args()
full_name = os.path.join(inputArgs.folder, inputArgs.file_name)


def checkCellOutput(fileName, expected_stdout):
    notebook = json.load(open(fileName, "r"))
    code_cells = (cell for cell in notebook["cells"] if cell["cell_type"] == "code")
    for cell, expected_output in zip(code_cells, expected_stdout):
        source = cell["source"]
        print("Checking cell starting with: " + source[0])
        for actual_output in cell["outputs"]:
            if "text" in actual_output:
                actual_output_text = actual_output["text"]
                for actual_line, expected_line in zip(
                    actual_output_text, expected_output
                ):
                    assert actual_line.startswith(expected_line), (
                        'Actual Line "'
                        + actual_line
                        + '" didn\'t match "'
                        + expected_line
                        + '"'
                    )
                assert len(actual_output_text) == len(expected_output), (
                    "Actual output length = "
                    + str(len(actual_output_text))
                    + ", expected_length - "
                    + str(len(expected_output))
                )
    print("checkCellOutput completed")


def checkSpecifiedCellOutput(fileName, cell_source, cell_output_substring):
    # assert that a specific code cell contains a substring (case insensitve)
    notebook = json.load(open(fileName, "r"))
    code_cells = (cell for cell in notebook["cells"] if cell["cell_type"] == "code")
    msg = (
        "actual output {} contain expected "
        "substring:\nactual output = {}\nexpected substring={}"
    )
    for cell in code_cells:
        source = cell["source"]
        if source != cell_source:
            continue
        print("Checking cell starting with: " + source[0])
        for actual_output in cell["outputs"]:
            actual_output_str = str(actual_output)
            bad_msg = msg.format("does not", actual_output_str, cell_output_substring)
            assert cell_output_substring.lower() in actual_output_str.lower(), bad_msg
    print("checkSpecifiedCellOutput completed")


def checkWidgetOutput(fileName):
    widget_property = "application/aml.mini.widget.v1"
    widget_data_found = False
    notebook = json.load(open(fileName, "r"))
    code_cells = (cell for cell in notebook["cells"] if cell["cell_type"] == "code")
    for cell in code_cells:
        for actual_output in cell["outputs"]:
            if "data" in actual_output:
                actual_output_data = actual_output["data"]
                if widget_property in actual_output_data:
                    print("Widget data found")
                    widget_data = actual_output_data[widget_property]
                    assert widget_data.startswith('{"status": "Completed"'), widget_data
                    print("Widget data valid")
                    widget_data_found = True
    assert widget_data_found
    print("checkWidgetOutput completed")


if inputArgs.expected_stdout is not None:
    checkCellOutput(full_name, inputArgs.expected_stdout)

if inputArgs.cell_source is not None:
    checkSpecifiedCellOutput(
        full_name, inputArgs.cell_source, inputArgs.cell_output_substring
    )

if inputArgs.check_widget:
    checkWidgetOutput(full_name)
