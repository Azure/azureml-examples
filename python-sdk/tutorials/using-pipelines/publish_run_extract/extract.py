import argparse
import os

print("In extract.py")
print("As a data scientist, this is where I use my extract code.")

parser = argparse.ArgumentParser("extract")
parser.add_argument("--input_extract", type=str, help="input_extract data")
parser.add_argument("--output_extract", type=str, help="output_extract directory")

args = parser.parse_args()

print("Argument 1: %s" % args.input_extract)
print("Argument 2: %s" % args.output_extract)

if not (args.output_extract is None):
    os.makedirs(args.output_extract, exist_ok=True)
    print("%s created" % args.output_extract)

with open(os.path.join(args.input_extract, "20news.pkl"), "rb") as f:
    content = f.read()
    with open(os.path.join(args.output_extract, "20news.pkl"), "wb") as fw:
        fw.write(content)
