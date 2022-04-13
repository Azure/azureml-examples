# imports
import contextlib
import imp
import os
import json
import glob
import argparse
from pathlib import Path


@contextlib.contextmanager
def change_working_dir(path):
    """Context manager for changing the current working directory"""

    saved_path = os.getcwd()
    os.chdir(str(path))
    try:
        yield
    finally:
        os.chdir(saved_path)

@contextlib.contextmanager
def replace_content(file, skip_wait=True, force_rerun=True):
    wait_str = "ml_client.jobs.stream(pipeline_job.name)"
    replace_holder = "## PLACEHOLDER"
    dsl_str = "@dsl.pipeline("
    rerun_str = "@dsl.pipeline(force_rerun=True,"

    with open(file, encoding="utf-8") as f:
        original_content = f.read()
    try:
        with open(file, "w") as f:
            new_content = original_content
            if skip_wait:
                new_content = new_content.replace(wait_str, replace_holder)
            if force_rerun:
                new_content = new_content.replace(dsl_str, rerun_str)
            f.write(new_content)
        yield
    finally:
        if skip_wait:
            with open(file, 'w') as f:
                f.write(original_content)


def main(args):
    
    # get list of notebooks
    notebooks = sorted(glob.glob("**/*.ipynb", recursive=True))

    for notebook in notebooks:
        notebook = Path(notebook)
        folder = notebook.parent
        with change_working_dir(folder), replace_content(notebook.name, args.skip_wait):
            os.system(f"papermill {notebook.name} out.ipynb -k python")


if __name__ == "__main__":
    # setup argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-wait", type=bool, default=True)
    args = parser.parse_args()

    # call main
    main(args)