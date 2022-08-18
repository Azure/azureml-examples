import argparse
import json
import os


def check_test_case(input_file):
    error_count = 0
    error_list = []
    not_run_list = []

    with open(input_file) as f:
        files = f.readlines()
        for file in files:
            file = file.replace("\n", "")
            if ".ipynb" in file:
                file = file.replace(".ipynb", ".output.ipynb")
                if not os.path.isfile(file):
                    not_run_list.append(file)
                    continue
                with open(file) as output_file:
                    output_file_obj = json.load(output_file)
                    if (
                        "An Exception was encountered at"
                        in output_file_obj["cells"][0]["source"][0]
                    ):
                        error_count += 1
                        error_list.append(file)

    if error_count != 0:
        for err in error_list:
            print(err)

        if len(not_run_list) > 0:
            print("\nThese test case are skipped")
            for not_run in not_run_list:
                print(not_run)

        raise Exception("Error occurs in these test cases")


if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Check all papermill output files.")
    parser.add_argument("-i", "--input", required=True, help="job ipynb file list")

    args = parser.parse_args()

    check_test_case(args.input)
