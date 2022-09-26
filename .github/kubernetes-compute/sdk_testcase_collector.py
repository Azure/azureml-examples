import argparse
import pathlib
import yaml
import re


def collect_test_cases(output_file, regex):
    root_dir = ".github/workflows"
    root = pathlib.Path(root_dir)

    testcases = []
    for item in root.iterdir():
        testcase_filename = str(item).split("/")[-1]
        # print(testcase_filename)
        if re.match(regex, testcase_filename) is not None:
            print(testcase_filename)
            # testcases.append(testcase_filename)
            yaml_stream = open(item)
            yaml_obj = yaml.load(yaml_stream, Loader=yaml.Loader)
            for step in yaml_obj["jobs"]["build"]["steps"]:
                if ".ipynb" in step["name"]:
                    work_dir = step["working-directory"]
                    notebook_name = step["name"].split("/")[-1]
                    testcases.append(f"{work_dir}/{notebook_name}\n")

    with open(output_file, "w") as f:
        f.writelines(testcases)


if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Collect all sdk test case with a regex."
    )
    parser.add_argument("-r", "--regex", required=True, help="test case name selector")
    parser.add_argument(
        "-o", "--output", required=False, help="the file selected test case send to"
    )

    args = parser.parse_args()

    collect_test_cases(args.output, args.regex)
