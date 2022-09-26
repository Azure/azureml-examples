import argparse
import os
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument("--input_data", type=str)
parser.add_argument("--output_folder", type=str)
args = parser.parse_args()

df = pd.read_csv(args.input_data)
print(df.head(10))

output_path = os.path.join(args.output_folder, "my_output.parquet")
df.to_parquet(output_path)
