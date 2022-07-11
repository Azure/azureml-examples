import argparse
import yaml
import os


def convert(input_file, compute_target, instance_type, common_runtime, output_file):
    def _convert(input_file, data, job_schema):
        # check job type
        is_pipeline_job = False
        is_sweep_job = False
        if "pipelineJob" in job_schema or "jobs" in data:
            is_pipeline_job = True
        if "sweepJob" in job_schema or data.get("type") == "sweep":
            is_sweep_job = True

        print("Job type: pipelineJob", is_pipeline_job, "sweepJob:", is_sweep_job)

        # change compute target
        if compute_target:
            data["compute"] = "azureml:%s" % compute_target
            if is_pipeline_job:
                settings = data.get("settings", {})
                settings["default_compute"] = "azureml:%s" % compute_target
                data["settings"] = settings

        # set instance type
        if not is_pipeline_job and instance_type:
            resources = data.get("resources", {})
            resources["instance_type"] = instance_type
            data["resources"] = resources

        for field in ["trial", "component"]:
            if field not in data:
                continue

            file_field = data[field]
            if not isinstance(file_field, str):
                continue

            if file_field.startswith("file:"):
                file_field = file_field.split(":", 1)[1]

            print("Found sub job spec:", file_field)
            dirname = os.path.dirname(input_file)
            convert(
                os.path.join(dirname, file_field),
                compute_target,
                instance_type,
                common_runtime,
                "",
            )

        if is_pipeline_job:
            jobs = data.get("jobs", {})
            for step in jobs:
                print("Found step:", step)
                _convert(input_file, jobs[step], "")
            return

    print("Processing file:", input_file)
    if not os.path.exists(input_file):
        print("Warning: File doesn't exist: ", input_file)
        return
    with open(input_file, "r") as f:
        data = yaml.load(f, Loader=yaml.FullLoader)
        job_schema = data.get("$schema", "")
        _convert(input_file, data, job_schema)

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
    parser.add_argument("-c", "--compute-target", required=False, help="Compute target")
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
