import argparse
import os.path
import pathlib

import ruamel.yaml as yaml

shared_steps_yml = """
    - name: check out repo
      uses: actions/checkout@v2
    - name: setup python
      uses: actions/setup-python@v2
      with:
        python-version: '3.8'
    - name: install tools
      run: bash .github/amlarc-tool.sh install_tools
      timeout-minutes: 30
    - name: azure login
      uses: azure/login@v1
      with:
        creds: ${{secrets.AZ_AE_CREDS}}
      timeout-minutes: 30
"""

workflow_inputs_yml = """
    inputs:
      SUBSCRIPTION:
        description: Subscription ID
        required: false
        default: 6560575d-fa06-4e7d-95fb-f962e74efd7a
      RESOURCE_GROUP:
        description: Resource Group
        required: false
        default: azureml-examples-rg
      WORKSPACE:
        description: Workspace Name
        required: false
        default: amlarc-githubtest-ws
"""

env_yml = """
      SUBSCRIPTION: ${{ github.event.inputs.SUBSCRIPTION }}
      RESOURCE_GROUP: ${{ github.event.inputs.RESOURCE_GROUP }}
      WORKSPACE: ${{ github.event.inputs.WORKSPACE }}
"""

shared_steps = yaml.round_trip_load(shared_steps_yml)
workflow_inputs = yaml.round_trip_load(workflow_inputs_yml)
env = yaml.round_trip_load(env_yml)


def convert(input_file):
    with open(input_file, "r") as f:
        data = yaml.round_trip_load(f)

        # add amlarc suffix
        data["name"] += "-amlarc"

        # add inputs
        data["on"]["workflow_dispatch"] = workflow_inputs

        # add env
        data["jobs"]["build"]["env"] = env

        # change job steps
        steps = data["jobs"]["build"]["steps"]
        new_step = shared_steps
        for single_step in steps:
            if single_step["name"] == "run job":
                command = single_step["run"].split(" ")
                target_file = command[-1]

                case_level = len(single_step.get("working-directory", "").split("/"))
                amlarc_tool_path = '.github/amlarc-tool.sh'
                for _ in range(case_level):
                    amlarc_tool_path = os.path.join("..", amlarc_tool_path)

                new_command = (
                    [f"bash {amlarc_tool_path} run_cli_job"]
                    + [target_file]
                    + ["-cr"]
                )
                single_step["run"] = " ".join(new_command)
                new_step.append(single_step)
            else:
                continue
        data["jobs"]["build"]["steps"] = new_step

        # modify the pull request trigger
        orig_paths = data["on"]["pull_request"]["paths"]
        new_paths = []
        for i in orig_paths:
            if ".github/workflows/cli-jobs" in i:
                t = i.split("/")
                tt = t[-1].split(".")
                tt = "%s-amlarc.yml" % tt[0]
                new_paths.append("/".join(t[:-1] + [tt]))
                continue
            if "sh" in i:
                continue
            new_paths.append(i)
        new_paths.append(".github/amlarc-tool.sh")
        data["on"]["pull_request"]["paths"] = new_paths

    # write back with suffix -amlarc
    output_file = os.path.join(
        os.path.dirname(input_file),
        "%s-amlarc.yml" % os.path.basename(input_file).split(".")[0],
    )
    with open(output_file, "w") as f:
        yaml.round_trip_dump(data, f, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # input file
    parser.add_argument("-i", "--input", type=str, required=True, help="input file")
    args = parser.parse_args()
    convert(args.input)
