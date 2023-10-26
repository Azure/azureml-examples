import argparse
import os


argParser = argparse.ArgumentParser()
argParser.add_argument("working_directory", help="The working directory for this script.")
argParser.add_argument("sample_path", help="The absolute path to the sample directory.")
args = argParser.parse_args()
working_directory = args.working_directory.replace("\\", "/")
sample_path = args.sample_path.replace("\\", "/")


def main():
    INVALID_README_MSG = f"{sample_path} does not contain a README.md file with all required words. See the Discoverability section of CONTRIBUTING.md."
    EXCLUSIONS_FILE_PATH = f"{working_directory}/infra/bootstrapping/readme_validation_exclusions.txt"
    required_sections = ["overview", "objective", "programming languages", "estimated runtime"]

    print(f"Checking if {sample_path} contains a README.md file with all required words...")

    # Check if sample is excluded from README validation
    with open(EXCLUSIONS_FILE_PATH, encoding="utf-8") as exclusions_file:
        exclusions = exclusions_file.read().splitlines()
        if sample_path in exclusions:
            print(f"Skipping {sample_path} since it is excluded from README validation.")
            return 0

    # Check if sample contains a valid README.md file
    try:
        with open(f"{sample_path}/README.md", encoding="utf-8") as readme_file:
            readme_content = readme_file.read()

            if all([section in readme_content for section in required_sections]):
                print(f"{sample_path} contains a README.md file with all required sections.")
                return 0
    except FileNotFoundError:
        pass

    print(INVALID_README_MSG)
    return 1


if __name__ == "__main__":
    main()
