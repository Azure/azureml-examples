import argparse
from pathlib import Path
from uuid import uuid4
from datetime import datetime
import os

parser = argparse.ArgumentParser()
parser.add_argument("--bool_input", type=bool, help="A boolean input")
args = parser.parse_args()
print(args.bool_input)
print(type(args.bool_input))