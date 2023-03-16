import pandas as pd
import argparse

parser = argparse.ArgumentParser()
# parse --data_file argument
parser.add_argument("--data_file", type=str)
# parse --train_file, --validation_file, --test_file arguments
parser.add_argument("--train_file", type=str)
parser.add_argument("--validation_file", type=str)
parser.add_argument("--test_file", type=str)
# parse --train_split, --validation_split, --test_split arguments
parser.add_argument("--train_split", type=float)
parser.add_argument("--validation_split", type=float)
parser.add_argument("--test_split", type=float)

#prase arguments
args = parser.parse_args()

print("input data file path: ", args.data_file)

# read args.data_file into a pandas dataframe from json lines file
df = pd.read_json(args.data_file, lines=True)
# split first train_split% of data into train set and write to args.train_file
df[:int(len(df)*args.train_split)].to_json(args.train_file, orient='records', lines=True)
# split next validation_split% of data into validation set and write to args.validation_file
df[int(len(df)*args.train_split):int(len(df)*(args.train_split+args.validation_split))].to_json(args.validation_file, orient='records', lines=True)
# split next test_split% of data into test set and write to args.test_file
df[int(len(df)*(args.train_split+args.validation_split)):int(len(df)*(args.train_split+args.validation_split+args.test_split))].to_json(args.test_file, orient='records', lines=True)

