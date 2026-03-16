# imports
import argparse
from pathlib import Path

import mlflow
import pandas as pd

parser = argparse.ArgumentParser()

parser.add_argument("--model", type=str)
parser.add_argument("--test_data", type=str)
parser.add_argument("--predict_result", type=str)

args = parser.parse_args()

test_data_dir = Path(args.test_data)
predict_result_dir = Path(args.predict_result)
predict_result_dir.mkdir(parents=True, exist_ok=True)

x_test_path = test_data_dir / "X_test.csv"
y_test_path = test_data_dir / "y_test.csv"
if not x_test_path.exists() or not y_test_path.exists():
	raise FileNotFoundError(
		f"Expected test data files not found in {test_data_dir}. "
		f"Found files: {[p.name for p in test_data_dir.glob('*')]}"
	)

X_test = pd.read_csv(x_test_path)
model = mlflow.sklearn.load_model(Path(args.model))
y_test = pd.read_csv(y_test_path)
y_test["predict"] = model.predict(X_test)

y_test.to_csv(predict_result_dir / "predict_result.csv", index=False)
