import argparse
import os
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument("--input_data", type=str)
args = parser.parse_args()

file_name = os.path.join(args.input_data, "titanic.csv")
df = pd.read_csv(file_name)
print(df.head(10))
