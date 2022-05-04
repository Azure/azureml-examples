import argparse
import os
import shutil

def read_file(file_path, arg_name):
    with open(file_path, "r") as f:
        content = f.readlines()
        print("{}: Reading from {} \n{}".format(arg_name, file_path, content))

parser = argparse.ArgumentParser("script")

parser.add_argument("--uri_file_input_mount", type=str)
parser.add_argument("--uri_folder_input_mount", type=str)
parser.add_argument("--mltable_eval_mount", type=str)
parser.add_argument("--mltable_eval_download", type=str)
parser.add_argument("--mltable_mount_input", type=str)
parser.add_argument("--mltable_download_input", type=str)
parser.add_argument("--mlflowmodel_mount_input", type=str)
parser.add_argument("--uri_file_input_download", type=str)
parser.add_argument("--uri_folder_input_download", type=str)

parser.add_argument("--uri_folder_input_direct", type=str)

parser.add_argument("--uri_folder_upload", type=str)
parser.add_argument("--uri_file_upload", type=str)
parser.add_argument("--uri_folder_mount", type=str)
parser.add_argument("--uri_file_mount", type=str)
parser.add_argument("--mltable_mount", type=str)
parser.add_argument("--mltable_upload", type=str)
parser.add_argument("--mlflow_model_mount", type=str)
parser.add_argument("--mlflow_model_upload", type=str)

args = parser.parse_args()
filename = "data.csv"

# direct mode
print("direct mode for input {}: {}".format('uri_folder_input_direct', args.uri_folder_input_direct))

#single file input
read_file(args.uri_file_input_mount, "roUriFile") # single file
read_file(args.uri_file_input_download, "downloadUriFile") # single file

# listing input folder
print("Listing dir from {} \n {}".format(args.uri_folder_input_mount, os.listdir(args.uri_folder_input_mount)))
print("Listing dir from {} \n {}".format(args.mltable_eval_mount, os.listdir(args.mltable_eval_mount)))
print("Listing dir from {} \n {}".format(args.mltable_eval_download, os.listdir(args.mltable_eval_download)))
print("Listing dir from {} \n {}".format(args.mlflowmodel_mount_input, os.listdir(args.mlflowmodel_mount_input)))
print("Listing dir from {} \n {}".format(args.uri_folder_input_download, os.listdir(args.uri_folder_input_download)))

read_file(os.path.join(args.uri_folder_input_mount, filename), "uri_folder_input_mount")
read_file(os.path.join(args.uri_folder_input_download, filename), "uri_folder_input_download")

# outputs
# copy artifact file
shutil.copy(src=os.path.join(args.mlflowmodel_mount_input, 'MLmodel'), dst=args.mlflow_model_mount)
shutil.copy(src=os.path.join(args.mlflowmodel_mount_input, 'MLmodel'), dst=args.mlflow_model_upload)

shutil.copy(src=os.path.join(args.mltable_mount_input, 'MLTable'), dst=args.mltable_mount)
shutil.copy(src=os.path.join(args.mltable_mount_input, 'MLTable'), dst=args.mltable_upload)

# output data
# with open(os.path.join(args.uri_folder_upload, filename), "w+") as f:
#     f.write("hello uri_folder_upload")
#     print("Wrote {} to {}".format(filename, args.uri_folder_upload))

with open(os.path.join(args.mltable_upload, filename), "w+") as f:
    f.write("hello mltable_upload")
    print("Wrote {} to {}".format(filename, args.mltable_upload))

with open(os.path.join(args.mltable_mount, filename), "w+") as f:
    f.write("hello mltable_mount")
    print("Wrote {} to {}".format(filename, args.mltable_mount))

with open(os.path.join(args.mlflow_model_mount, filename), "w+") as f:
    f.write("hello mlflow_model_mount")
    print("Wrote {} to {}".format(filename, args.mlflow_model_mount))

with open(os.path.join(args.mlflow_model_upload, filename), "w+") as f:
    f.write("hello mlflow_model_upload")
    print("Wrote {} to {}".format(filename, args.mlflow_model_upload))

with open(os.path.join(args.uri_folder_mount, filename), "w+") as f:
    f.write("hello uri_folder_mount")
    print("Wrote {} to {}".format(filename, args.uri_folder_mount))

with open(args.uri_file_upload, "w+") as f:
    f.write("hello uri_file_upload")
    print("Wrote hello to {}".format(args.uri_file_upload))

with open(args.uri_file_mount, "w+") as f:
    f.write("hello uri_file_mount")
    print("Wrote hello to {}".format(args.uri_file_mount))
