import argparse
import pandas as pd
from datasets import load_dataset

test_data = "./small-test.jsonl"

parser = argparse.ArgumentParser()
parser.add_argument("--test-data", type=str, default=test_data, dest="test_data_file")
args = parser.parse_args()

hf_test_data = load_dataset("cnn_dailymail", "3.0.0", split="test", streaming=True)

test_data_df = pd.DataFrame(hf_test_data.take(1000))
test_data_df["input_string"] = test_data_df["article"]
test_data_df["summary"] = test_data_df["highlights"]

test_data_df.to_json(args.test_data_file, lines=True, orient="records")
