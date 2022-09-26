import argparse
import os
import tarfile

parser = argparse.ArgumentParser()
parser.add_argument("--input_data", type=str)
parser.add_argument("--output_folder", type=str)
args = parser.parse_args()


file = tarfile.open(args.input_data)
output_path = os.path.join(args.output_folder)
file.extractall(output_path)  
file.close()
