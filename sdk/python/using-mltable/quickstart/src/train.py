import argparse
import mltable

# parse arguments
parser = argparse.ArgumentParser()
parser.add_argument("--input", help="mltable to read")
args = parser.parse_args()

# load mltable
tbl = mltable.load(args.input)

# load into pandas
df = tbl.to_pandas_dataframe()

# print the head of data
print(df.head())
