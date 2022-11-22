import argparse
import os
from datetime import datetime

print("Hello Python World")

parser = argparse.ArgumentParser()
parser.add_argument("--input_data", type=str)
parser.add_argument("--input_string", type=str)
parser.add_argument("--output_data", type=str)

args = parser.parse_args()

print("sample_input_string: %s" % args.input_string)
print("sample_input_data path: %s" % args.input_data)
print("sample_output_data path: %s" % args.output_data)

# with open(os.path.join(args.input_data, "hello-world.txt"), "wt") as text_file:
# print("hello world inputs", file=text_file)

with open(os.path.join(args.output_data, "hello-world.txt"), "wt") as text_file:
    print("hello world", file=text_file)
