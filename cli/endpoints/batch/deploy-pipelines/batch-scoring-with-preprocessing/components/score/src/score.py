import argparse
import os
import glob
from pathlib import Path
import mlflow
import pandas as pd

parser = argparse.ArgumentParser("score")
parser.add_argument("--model_path", type=str, help="Path of input model")
parser.add_argument("--data_path", type=str, help="Path to test data")
parser.add_argument("--score_mode", type=str, help="The mode of the prediction. `append` or `prediction_only`.")
parser.add_argument("--scores_path", type=str, help="Path of predictions")

args = parser.parse_args()

lines = [
    f"Model path: {args.model_path}",
    f"Input data path: {args.data_path}",
    f"Predictions path: {args.scores_path}",
]

for line in lines:
    print(line)

print("Loading model")
model = mlflow.pyfunc.load_model(args.model_path)

print(f"Reading all the CSV files from path {args.data_path}")
input_files = glob.glob(args.data_path + "/*.csv")

for input_file in input_files:
    print(f"Working on file {input_file}")
    df = pd.read_csv(input_file, dtype={ "ca":"Int64", "cp": "Int64", "exang": "Int64", "fbs": "Int64", "restecg": "Int64", "sex": "Int64", "slope": "Int64", "thal": "Int64" })
    print(df.dtypes)
    print(model.metadata.get_input_schema())
    predictions = model.predict(df)

    if args.score_mode == 'append':
        df['prediction'] = predictions
    else:
        df = pd.DataFrame(predictions, columns=['prediction'])

    output_file_name = Path(input_file).stem
    output_file_path = os.path.join(args.scores_path, output_file_name + '.csv')
    print(f"Writing file {output_file_path}")
    df.to_csv(output_file_path, index=False)
