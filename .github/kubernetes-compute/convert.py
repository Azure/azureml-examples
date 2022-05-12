import argparse
import yaml


def convert(input_file, compute_target, instance_type, common_runtime, output_file):
    with open(input_file, "r") as f:
        data = yaml.load(f, Loader=yaml.FullLoader)
        job_schema = data.get("$schema", "")
        is_pipeline_job = False
        is_sweep_job = False
        if "pipelineJob" in job_schema or "jobs" in data:
            is_pipeline_job = True
        if "sweepJob" in job_schema or data.get("type") == "sweep":
            is_sweep_job = True

        # change compute target
        data["compute"] = "azureml:%s" % compute_target
        if is_pipeline_job:
            settings = data.get("settings", {})
            settings["default_compute"] = "azureml:%s" % compute_target
            data["settings"] = settings

            for step in data.get("jobs", {}):
                data["jobs"][step]["compute"] = "azureml:%s" % compute_target

        # set instance type
        if not is_pipeline_job and instance_type:
            resources = data.get("resources", {})
            resources["instance_type"] = instance_type
            data["resources"] = resources

        # set common runtime environment variables.
        if common_runtime:
            if is_pipeline_job:
                for step in data.get("jobs", {}):
                    env = data["jobs"][step].get("environment_variables", {})
                    env["AZUREML_COMPUTE_USE_COMMON_RUNTIME"] = "true"
                    data["jobs"][step]["environment_variables"] = env
            elif is_sweep_job:
                env = data["trial"].get("environment_variables", {})
                env["AZUREML_COMPUTE_USE_COMMON_RUNTIME"] = "true"
                data["trial"]["environment_variables"] = env
            else:
                env = data.get("environment_variables", {})
                env["AZUREML_COMPUTE_USE_COMMON_RUNTIME"] = "true"
                data["environment_variables"] = env

        # write to output file if output file is specified, otherwise change inplace.
        if output_file:
            with open(output_file, "w") as f:
                yaml.dump(data, f)
        else:
            with open(input_file, "w") as f:
                yaml.dump(data, f)


if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Convert test case to AMLARC-compatible files."
    )
    parser.add_argument("-i", "--input", required=True, help="Input test case file")
    parser.add_argument(
        "-o",
        "--output",
        required=False,
        help="Output AMLARC-compatible file, if not provides, " "replace file inplace",
    )
    parser.add_argument(
        "-c",
        "--compute-target",
        required=False,
        help='Compute target, default is "githubtest"',
        default="githubtest",
    )
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
    convert(
        args.input,
        args.compute_target,
        args.instance_type,
        args.common_runtime,
        args.output,
    )
