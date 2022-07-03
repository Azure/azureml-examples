# imports
import contextlib
import os
import glob
import argparse
import concurrent.futures
import traceback

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
    dsl_str = "@pipeline("
    rerun_str = "@pipeline(force_rerun=True,"

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
        with open(file, "w", encoding="utf-8") as f:
            f.write(original_content)


def run_notebook(notebook, skip_wait=True, force_rerun=True, dry_run=False):
    notebook = Path(notebook)
    folder = notebook.parent
    with change_working_dir(folder), replace_content(
        notebook.name, skip_wait, force_rerun
    ):
        command = f"papermill {notebook.name} out.ipynb -k python"
        print(f"Running {command}")
        if not dry_run:
            os.system(command)


def main(args):

    # get list of notebooks
    notebooks = sorted(glob.glob("**/*.ipynb", recursive=True))
    with concurrent.futures.ProcessPoolExecutor(max_workers=12) as executor:
        future_to_notebooks = {
            executor.submit(
                run_notebook, notebook, args.skip_wait, args.force_rerun, args.dry_run
            ): notebook
            for notebook in notebooks
            if "out.ipynb" not in notebook
        }

        for future in concurrent.futures.as_completed(future_to_notebooks):
            notebook = future_to_notebooks[future]
            try:
                future.result()
            except BaseException as exc:
                print(f"Failed to run {notebook} due to {exc}")
                raise exc


if __name__ == "__main__":
    # setup argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-wait", type=bool, default=True)
    parser.add_argument("--force-rerun", type=bool, default=True)
    parser.add_argument("--dry-run", type=bool, default=False)

    args = parser.parse_args()

    # call main
    main(args)
