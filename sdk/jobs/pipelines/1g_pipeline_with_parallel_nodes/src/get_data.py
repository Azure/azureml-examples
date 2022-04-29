
import argparse
import os
from pathlib import Path

print ("Get file and tabular data")

parser = argparse.ArgumentParser()
parser.add_argument("--input_data", type=str)
parser.add_argument("--file_output_data", type=str)
parser.add_argument("--tabular_output_data", type=str)

args = parser.parse_args()

print("sample_input_data path: %s" % args.input_data)
print("sample_file_output_data path: %s" % args.file_output_data)
print("sample_tabular_output_data path: %s" % args.tabular_output_data)

print("files in input_data path: ")
arr = os.listdir(args.input_data)
print(arr)

for folder_name in arr:
    if folder_name == "mnist-data":
        data_path = args.file_output_data + "/" + folder_name
        files = os.listdir(data_path)
        output_dir = Path(args.file_output_data)
        print("file_output_dir", output_dir)
        print("file_output_dir exits", Path(output_dir).exists())

        for file_path in files:
            file = Path(file_path)
            print("Processing {}".format(file))
            assert file.exists()
            (Path(output_dir) / file.name).write_text(file_path)
    elif folder_name == "iris-mltable":
        data_path = args.tabular_output_data + "/" + folder_name
        files = os.listdir(data_path)
        output_dir = Path(args.tabular_output_data)
        print("tabular_output_dir", output_dir)
        print("tabular_output_dir exits", Path(output_dir).exists())

        for file_path in files:
            file = Path(file_path)
            print("Processing {}".format(file))
            assert file.exists()
            (Path(output_dir) / file.name).write_text(file_path)
    else:
        pass


