import os
import argparse
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument("--data-path")
args = parser.parse_args()

print("Hello World!")

df = pd.read_csv(args.data_path)
print(df.head())