import os
import json
import glob
import argparse
import re
import subprocess
import random
import sys


def get_all_files(path, valid_suffix):
    """
    Get all files in a directory with a certain suffix
    """
    files = []
    for suffix in valid_suffix:
        files.extend(glob.glob(path + "*/**/*" + suffix, recursive=True))
    return files


class Job:
    RANDOM_INT = random.randint(0, 1000000)

    def __init__(self, pipeline_path):
        self._pipeline_path = os.path.abspath(pipeline_path)

    @property
    def pipeline_path(self):
        return self._pipeline_path

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

    def update_script(self):
        for script in self.scripts:
            with open(script, "r") as f:
                content = f.read()
            if script.endswith(".py"):
                content += f"\nprint(\"{self.RANDOM_INT}\")\n"
            elif script.endswith(".R"):
                content += f"\nprint(\"{self.RANDOM_INT}\")\n"
            else:
                content = content.replace("echo", f"echo {self.RANDOM_INT} & echo")
            with open(script, "w") as f:
                f.write(content)

    def recover_script(self):
        for script in self.scripts:
            with open(script, "r") as f:
                content = f.read()
            if script.endswith(".py") or script.endswith(".R"):
                content = re.sub(f"\nprint\\(\"[0-9]+\"\\)\n", "", content)
            else:
                while True:
                    next_content = re.sub("echo [0-9]+ & echo", "echo", content)
                    if next_content == content:
                        break
                    content = next_content
            with open(script, "w") as f:
                f.write(content)


def main():
    # get list of jobs
    jobs = list(map(lambda x: Job(x), get_all_files(os.path.join(os.path.dirname(__file__), "jobs", "pipeline"), ["pipeline.yml", "pipeline.yaml"])))
    print(len(jobs), "pipelines found")
    for job in jobs:
        if sys.argv[1] == "update":
            job.update_script()
        elif sys.argv[1] == "recover":
            job.recover_script()


if __name__ == "__main__":
    main()
