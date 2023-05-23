import shutil
import argparse
import os

parser = argparse.ArgumentParser("register_model")
parser.add_argument("--model_input", type=str, help="Path to input model data")
parser.add_argument(
    "--evaluation_input", type=str, help="Path to input evaluation result data"
)
parser.add_argument(
    "--model_output", type=str, help="Path of output model to be registered"
)

args = parser.parse_args()

for file_name in os.listdir(args.model_input):
    source = os.path.join(args.model_input, file_name)
    destination = os.path.join(args.model_output, file_name)

    if os.path.isdir(source):
        shutil.copytree(source, destination)
    else:
        shutil.copy(source, destination)
