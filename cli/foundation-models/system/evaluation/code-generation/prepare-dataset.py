import argparse
import pandas as pd
from datasets import load_dataset

test_data = "./small-test.jsonl"

parser = argparse.ArgumentParser()
parser.add_argument("--test-data", type=str, default=test_data, dest="test_data_file")
args = parser.parse_args()

hf_test_data = load_dataset("openai_humaneval", split="test")

test_data_df = pd.DataFrame(hf_test_data)

test_data_df.to_json(args.test_data_file, lines=True, orient="records")
