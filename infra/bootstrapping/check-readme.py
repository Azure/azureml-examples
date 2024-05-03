import argparse
import os
import sys

from pathlib import Path


argParser = argparse.ArgumentParser()
argParser.add_argument(
    "sample_path", help="The absolute path to the sample directory.", type=Path
)
args = argParser.parse_args()
sample_path = str(args.sample_path).strip()
repo_root = Path(__file__).resolve().parent.parent


def main():
    INVALID_README_MSG = f"{sample_path} does not contain a README.md file with all required words. See the Discoverability section of CONTRIBUTING.md to create a valid README or add the file path to ./infra/bootstrapping/readme_validation_exclusions.txt."
    EXCLUSIONS_FILE_PATH = f"{repo_root}/bootstrapping/readme_validation_exclusions.txt"
    required_sections = [
        "overview",
        "objective",
        "programming languages",
        "estimated runtime",
        "page_type: sample",
        "languages:",
        "products:",
        "description:",
    ]

    print(
        f"Checking if {sample_path} contains a README.md file with all required words..."
    )

    # Check if sample is excluded from README validation
    try:
        with open(EXCLUSIONS_FILE_PATH, encoding="utf-8") as exclusions_file:
            exclusions = exclusions_file.read().splitlines()

            for exclusion in exclusions:
                print(exclusion + "\n")

            if sample_path in exclusions:
                print(
                    f"Skipping {sample_path} since it is excluded from README validation."
                )
                sys.exit(0)

        # Check if sample contains a valid README.md file
        with open(f"{sample_path}/README.md", encoding="utf-8") as readme_file:
            readme_content = readme_file.read()
            if all(section in readme_content for section in required_sections):
                print(
                    f"{sample_path} contains a README.md file with all required sections."
                )
                sys.exit(0)
    except FileNotFoundError as e:
        print(f"hit error {e}")

    print(INVALID_README_MSG)
    sys.exit(1)


if __name__ == "__main__":
    main()
