import argparse
import os

def read_file(file_path, arg_name):
    with open(file_path, "r") as f:
        content = f.readlines()
        print("{}: Reading from {} \n{}".format(arg_name, file_path, content))
    # try:
    #     with open(file_path, "r") as f:
    #         content = f.readlines()
    #     print("{}: Reading from {} \n{}".format(arg_name, file_path, content))

    # except Exception as e:
    #     print("{}: Exception {}".format(arg_name, e))

parser = argparse.ArgumentParser("script")

parser.add_argument("--uri_file_input_mount", type=str)
parser.add_argument("--uri_folder_input_mount", type=str)
parser.add_argument("--uri_file_input_download", type=str)
parser.add_argument("--uri_folder_input_download", type=str)

parser.add_argument("--uri_folder_upload", type=str)
parser.add_argument("--uri_file_upload", type=str)
parser.add_argument("--uri_folder_mount", type=str)
parser.add_argument("--uri_file_mount", type=str)

parser.add_argument("--mlflow_model", type=str)
parser.add_argument("--mltable_eval_mount", type=str)

args = parser.parse_args()
filename = "data.csv"

read_file(args.uri_file_input_mount, "roUriFile") # single file
read_file(args.uri_file_input_download, "downloadUriFile") # single file

print("Listing dir from {} \n {}".format(args.uri_folder_input_mount, os.listdir(args.uri_folder_input_mount))) # write to input uri

read_file(os.path.join(args.uri_folder_input_mount, filename), "uri_folder_input_mount")
read_file(os.path.join(args.uri_folder_input_download, filename), "uri_folder_input_download")

try:
    print("Listing dir from {} \n {}".format(args.mltable_eval_mount, os.listdir(args.mltable_eval_mount))) # write to input uri
except Exception as e:
    print("mltable_eval_mount: Exception {}".format(e))

with open(os.path.join(args.mlflow_model, filename), "w+") as f:
    f.write("hello mlflow_model")
    print("Wrote {} to {}".format(filename, args.mlflow_model))

with open(os.path.join(args.uri_folder_upload, filename), "w+") as f:
    f.write("hello uri_folder_upload")
    print("Wrote {} to {}".format(filename, args.uri_folder_upload))

with open(os.path.join(args.uri_folder_mount, filename), "w+") as f:
    f.write("hello uri_folder_mount")
    print("Wrote {} to {}".format(filename, args.uri_folder_mount))

with open(args.uri_file_upload, "w+") as f:
    f.write("hello uri_file_upload")
    print("Wrote hello to {}".format(args.uri_file_upload))

with open(args.uri_file_mount, "w+") as f:
    f.write("hello uri_file_mount")
    print("Wrote hello to {}".format(args.uri_file_mount))
