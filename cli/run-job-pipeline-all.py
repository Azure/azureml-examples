import os
import json
import glob
import argparse
import re
import subprocess
import random
import sys
from tkinter.messagebox import NO
from typing import List


def get_all_files(path, valid_suffix):
    """
    Get all files in a directory with a certain suffix
    """
    files = []
    for suffix in valid_suffix:
        files.extend(glob.glob(path + "*/**/*" + suffix, recursive=True))
    return files


class Job:
    def __init__(self, pipeline_path):
        self._pipeline_path = pipeline_path

    @property
    def pipeline_path(self):
        return self._pipeline_path

    @property
    def pipeline_path_to_write(self):
        return "./" + self.pipeline_path.replace("\\", "/")

    @property
    def name(self):
        return os.path.basename(self.pipeline_path)

    @property
    def directory(self):
        return os.path.dirname(self.pipeline_path)

    @property
    def scripts(self):
        scripts = get_all_files(self.directory, [".py", ".R"])
        if len(scripts) == 0:
            scripts = get_all_files(self.directory, ["component.yml"])
            assert len(scripts) > 0, "No scripts found in " + self.directory
        return scripts

    def update_script(self, random_value):
        for script in self.scripts:
            with open(script, "r") as f:
                content = f.read()
            if script.endswith(".py"):
                content += f'\nprint("{random_value}")\n'
            elif script.endswith(".R"):
                content += f'\nprint("{random_value}")\n'
            else:
                content = content.replace("echo", f"echo {random_value} & echo")
            with open(script, "w") as f:
                f.write(content)

    def recover_script(self):
        for script in self.scripts:
            with open(script, "r") as f:
                content = f.read()
            if script.endswith(".py") or script.endswith(".R"):
                content = re.sub(f'\nprint\\("[0-9]+"\\)\n', "", content)
            else:
                while True:
                    next_content = re.sub("echo [0-9]+ & echo", "echo", content)
                    if next_content == content:
                        break
                    content = next_content
            with open(script, "w") as f:
                f.write(content)

    def get_run_shell(self, experiment_name=None) -> str:
        # return "az ml job create --file {}{}".format(
        #     self.pipeline_path_to_write,
        #     f" --set experiment_name={experiment_name}" if experiment_name else "",
        # )
        return "echo {0}\nbash run-job.sh {0}{1}".format(
            self.pipeline_path_to_write,
            f" {experiment_name} nowait" if experiment_name else "",
        )

    def get_run_and_wait_shell(self, experiment_name=None) -> str:
        return "echo {0}\nbash run-job.sh {0}{1}".format(
            self.pipeline_path_to_write,
            f" {experiment_name}" if experiment_name else "",
        )


class JobSet:
    def __init__(self, jobs: List[Job], random_value: str = None) -> None:
        self._random_value = random_value
        self.jobs = jobs

    @property
    def random_value(self):
        if self._random_value is None:
            return "$target_version"
        else:
            return self._random_value

    def update_script(self):
        for job in self.jobs:
            job.update_script(self.random_value)

    def recover_script(self):
        for job in self.jobs:
            job.recover_script()

    @property
    def create_dependency_shell(self) -> str:
        return """az ml compute create -n cpu-cluster --type amlcompute --min-instances 0 --max-instances 8 -o none
az ml compute create -n gpu-cluster --type amlcompute --min-instances 0 --max-instances 4 --size Standard_NC12 -o none
az ml data create --file assets/data/local-folder.yml --set version={0} -o none
az ml component create --file jobs/pipelines-with-components/basics/1b_e2e_registered_components/train.yml --set version={0} -o none
az ml component create --file jobs/pipelines-with-components/basics/1b_e2e_registered_components/score.yml --set version={0} -o none
az ml component create --file jobs/pipelines-with-components/basics/1b_e2e_registered_components/eval.yml --set version={0} -o none
az ml data create --file jobs/pipelines-with-components/rai_pipeline_adult_analyse/data/data_adult_test.yaml --set version={0} -o none
az ml data create --file jobs/pipelines-with-components/rai_pipeline_adult_analyse/data/data_adult_train.yaml --set version={0} -o none
az ml environment create --file jobs/pipelines-with-components/rai_pipeline_adult_analyse/environment/responsibleai-environment.yaml --set version={0} -o none""".format(
            self.random_value
        )

    def generate_run_all_shell(self, target_path) -> str:
        experiment_name = f"cli_samples_v2_{self.random_value}"
        shells = [
            """
if [ -z "$1" ]
  then
    target_version="$RANDOM"
  else
    target_version=$1
fi""",
            self.create_dependency_shell,
        ]
        shells.extend(map(lambda x: x.get_run_shell(experiment_name), self.jobs))
        shells[-1] = self.jobs[-1].get_run_and_wait_shell(experiment_name)
        shells.append("az --version")

        with open(target_path, "w", encoding="utf-8") as run_all_shell_file:
            run_all_shell_file.write("\n\n".join(shells))


def main():
    if len(sys.argv) >= 3:
        random_value = sys.argv[2]
    else:
        random_value = None

    # get list of jobs
    jobs = list(
        map(
            lambda x: Job(x),
            get_all_files(
                os.path.join(os.path.dirname(__file__), "jobs", "basics"),
                ["hello-pipeline*.yml"],
            ),
        )
    )
    jobs.extend(
        map(
            lambda x: Job(x),
            get_all_files(
                os.path.join(os.path.dirname(__file__), "jobs", "pipeline"),
                ["pipeline.yml", "pipeline.yaml"],
            ),
        )
    )
    print(len(jobs), "pipelines found")
    job_set = JobSet(jobs, random_value)

    if sys.argv[1] == "update":
        job_set.update_script()
    elif sys.argv[1] == "recover":
        job_set.recover_script()
    elif sys.argv[1] == "generate":
        job_set.generate_run_all_shell("run-job-pipeline-all.sh")


if __name__ == "__main__":
    main()
