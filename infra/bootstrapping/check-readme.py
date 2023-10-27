import argparse
import os
import sys

from pathlib import Path


argParser = argparse.ArgumentParser()
argParser.add_argument("sample_path", help="The absolute path to the sample directory.", type=Path)
args = argParser.parse_args()
sample_path = args.sample_path
working_directory = Path(__file__).parent.parent


def main():
    INVALID_README_MSG = f"{sample_path} does not contain a README.md file with all required words. See the Discoverability section of CONTRIBUTING.md."
    EXCLUSIONS_FILE_PATH = f"{working_directory}/bootstrapping/readme_validation_exclusions.txt"
    required_sections = ["overview", "objective", "programming languages", "estimated runtime", "page_type: sample", "languages:", "products:", "description:"]

    print(f"Checking if {sample_path} contains a README.md file with all required words...")

    if Path(EXCLUSIONS_FILE_PATH).exists():
        # Check if sample is excluded from README validation
        with open(EXCLUSIONS_FILE_PATH, encoding="utf-8") as exclusions_file:
            exclusions = exclusions_file.read().splitlines()
            if sample_path in exclusions:
                print(f"Skipping {sample_path} since it is excluded from README validation.")
                sys.exit(0)
            else:
                print(f"{sample_path} is not excluded from README validation.")
                print(f"{sample_path} not in {exclusions}")
    else:
        print(f"{EXCLUSIONS_FILE_PATH} does not exist.")
        print(f"current working directory is {os.getcwd()}")

    # Check if sample contains a valid README.md file
    try:
        with open(f"{sample_path}/README.md", encoding="utf-8") as readme_file:
            readme_content = readme_file.read()

            if all(section in readme_content for section in required_sections):
                print(f"{sample_path} contains a README.md file with all required sections.")
                sys.exit(0)
    except FileNotFoundError:
        pass

    print(INVALID_README_MSG)
    sys.exit(1)


if __name__ == "__main__":
    main()
