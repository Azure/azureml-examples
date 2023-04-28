import argparse
import os
import glob
from pathlib import Path
import mlflow
import pandas as pd

parser = argparse.ArgumentParser("score")
parser.add_argument("--model_path", type=str, help="Path to the input model")
parser.add_argument("--data_path", type=str, help="Path to the data to score")
parser.add_argument("--score_mode", type=str, help="The scoring mode. Possible values are `append` or `prediction_only`.")
parser.add_argument("--scores_path", type=str, help="Path of predictions")

args = parser.parse_args()

lines = [
    f"model path: {args.model_path}",
    f"scoring mode: {args.score_mode}",
    f"input data path: {args.data_path}",
    f"ouputs data path: {args.scores_path}",
]

for line in lines:
    print(f"/t{line}")

print("Loading model")
model = mlflow.pyfunc.load_model(args.model_path)
model_input_types = dict(zip(model.metadata.signature.inputs.input_names(), model.metadata.signature.inputs.pandas_types()))

print("Input schema:")
print(model.metadata.get_input_schema())

print(f"Reading all the CSV files from path {args.data_path}")
input_files = glob.glob(args.data_path + "/*.csv")

for input_file in input_files:
    print(f"Working on file {input_file}")
    df = pd.read_csv(input_file).astype(model_input_types)
    predictions = model.predict(df)

    if args.score_mode == 'append':
        df["prediction"] = predictions
    else:
        df = pd.DataFrame(predictions, columns=["prediction"])

    output_file_name = Path(input_file).stem
    output_file_path = os.path.join(args.scores_path, output_file_name + '.csv')
    print(f"Writing file {output_file_path}")
    df.to_csv(output_file_path, index=False)
