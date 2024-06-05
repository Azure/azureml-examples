import argparse
import pandas as pd
from datasets import load_dataset

test_data_mask_1 = "./small-test-[MASK].jsonl"  # [MASK]
# test_data_mask_2 = "./small-test-mask.jsonl"  # <mask>

parser = argparse.ArgumentParser()
parser.add_argument(
    "--test-data", type=str, default=test_data_mask_1, dest="test_data_file"
)
args = parser.parse_args()

hf_test_data = load_dataset(
    "rcds/wikipedia-for-mask-filling", "original_512", split="train", streaming=True
)

test_data_df = pd.DataFrame(hf_test_data.take(1000))
test_data_df["input_string"] = test_data_df["texts"]
test_data_df["title"] = test_data_df["masks"].apply(
    lambda x: x[0] if len(x) > 0 else ""
)

# test_data_mask_2_df = test_data_df
test_data_mask_1_df = pd.DataFrame(test_data_df)
test_data_mask_1_df["input_string"] = test_data_mask_1_df["input_string"].apply(
    lambda x: x.replace("<mask>", "[MASK]")
)

test_data_mask_1_df.to_json(args.test_data_file, lines=True, orient="records")
# test_data_mask_2_df.to_json(test_data_mask_2, lines=True, orient="records")
