import argparse
import os
from datetime import datetime

print("Hello Python World")

parser = argparse.ArgumentParser()
parser.add_argument("--input_data", type=str)
parser.add_argument("--placeholder", type=str)
parser.add_argument("--output_data", type=str)

args = parser.parse_args()

print("input_data: %s" % args.input_data)
print("placeholder path: %s" % args.placeholder)
print("sample_output_data path: %s" % args.output_data)

print("file in input_data path: ")
print(args.input_data)
with open(args.input_data, "r") as f:
    print(f.read())


cur_time_str = datetime.now().strftime("%b-%d-%Y-%H-%M-%S")

print(
    "Writing file: %s" % os.path.join(args.output_data, "file-" + cur_time_str + ".txt")
)
with open(
    os.path.join(args.output_data, "file-" + cur_time_str + ".txt"), "wt"
) as text_file:
    print(f"Logging date time: {cur_time_str}", file=text_file)
