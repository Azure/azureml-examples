# imports
import os
import json
import glob
import argparse

from git import Repo

# define functions
def main(args):
    # get list of scripts
    scripts = sorted(glob.glob("*.sh", recursive=False))

    # get list of changes files
    repo = Repo(search_parent_directories=True)
    changed_files = [
        f.a_path for f in repo.index.diff("origin/main") if "cli/" in f.a_path
    ]
    print(changed_files)


# run functions
if __name__ == "__main__":
    # issue #146
    if "posix" not in os.name:
        print(
            "windows is not supported, see issue #146 (https://github.com/Azure/azureml-examples/issues/146)"
        )
        exit(1)

    # setup argparse
    parser = argparse.ArgumentParser()
    args = parser.parse_args()

    # call main
    main(args)
