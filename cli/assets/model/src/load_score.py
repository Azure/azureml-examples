import argparse
import pandas as pd
import mlflow.sklearn
import pandas as pd
import json
import os

parser = argparse.ArgumentParser()
parser.add_argument("--input_data", type=str)
parser.add_argument("--input_model", type=str)
parser.add_argument("--output_folder", type=str)
args = parser.parse_args()

with open(args.input_data) as f:
   sample_data = json.load(f)

f.close()

#Print Data to output
print(sample_data)

sk_model = mlflow.sklearn.load_model(args.input_model)
predictions = sk_model.predict(sample_data["data"])

# Writing to stdout
print(predictions)

with open(os.path.join(args.output_folder,"predictions.txt"), 'x') as output:
    # Writing data to a file
    output.write(str(predictions))
output.close()

