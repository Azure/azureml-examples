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

def _resolve_file(base_dir: Path, filename: str) -> Path:
    direct_path = base_dir / filename
    if direct_path.exists():
        return direct_path

    matches = sorted(base_dir.rglob(filename))
    if matches:
        return matches[0]

    raise FileNotFoundError(
        f"Expected file '{filename}' not found under {base_dir}. "
        f"Top-level contents: {[p.name for p in base_dir.glob('*')]}"
    )


def _resolve_model_uri(model_dir: Path) -> str:
    if (model_dir / "MLmodel").exists():
        return model_dir.as_posix()

    mlmodel_matches = sorted(model_dir.rglob("MLmodel"))
    if mlmodel_matches:
        return mlmodel_matches[0].parent.as_posix()

    raise FileNotFoundError(
        f"Could not find MLmodel under {model_dir}. "
        f"Top-level contents: {[p.name for p in model_dir.glob('*')]}"
    )


x_test_path = _resolve_file(test_data_dir, "X_test.csv")
y_test_path = _resolve_file(test_data_dir, "y_test.csv")
model_uri = _resolve_model_uri(Path(args.model))

X_test = pd.read_csv(x_test_path)
model = mlflow.sklearn.load_model(model_uri)
y_test = pd.read_csv(y_test_path)
y_test["predict"] = model.predict(X_test)

y_test.to_csv(predict_result_dir / "predict_result.csv", index=False)
