import argparse
from pathlib import Path
from uuid import uuid4
from datetime import datetime
import os

def str_to_bool(str):
    try:
        str = str.lower()
        if str == 'true':
            return True
        elif str == 'false':
            return False
        else:
            return None
    except AttributeError as e:
        return None


parser = argparse.ArgumentParser()
parser.add_argument("--bool_input", type=str_to_bool, help="A boolean input")
args = parser.parse_args()
print(args.bool_input)
print(type(args.bool_input))