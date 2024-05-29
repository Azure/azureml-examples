import os
import glob
import mlflow
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List
from datasets import load_dataset

DATA_READERS = {
    ".csv": "csv",
    ".tsv": "tsv",
    ".parquet": "parquet",
    ".json": "json",
    ".jsonl": "json",
    ".arrow": "arrow",
    ".txt": "text",
}


def init():
    global model
    global output_file
    global task_name
    global text_column

    # AZUREML_MODEL_DIR is the path where the model is located.
    # If the model is MLFlow, you don't need to indicate further.
    model_path = glob.glob(os.environ["AZUREML_MODEL_DIR"] + "/*/")[0]
    # AZUREML_BI_TEXT_COLUMN is an environment variable you can use
    # to indicate over which column you want to run the model on. It can
    # used only if the model has one single input.
    text_column = os.environ.get("AZUREML_BI_TEXT_COLUMN", None)

    model = mlflow.pyfunc.load_model(model_path)
    model_info = mlflow.models.get_model_info(model_path)

    if not mlflow.openai.FLAVOR_NAME in model_info.flavors:
        raise ValueError(
            "The indicated model doesn't have an OpenAI flavor on it. Use "
            "``mlflow.openai.log_model`` to log OpenAI models."
        )

    if text_column:
        if (
            model.metadata
            and model.metadata.signature
            and len(model.metadata.signature.inputs) > 1
        ):
            raise ValueError(
                "The model requires more than 1 input column to run. You can't use "
                "AZUREML_BI_TEXT_COLUMN to indicate which column to send to the model. Format your "
                f"data with columns {model.metadata.signature.inputs.input_names()} instead."
            )

    task_name = model._model_impl.model["task"]
    output_path = os.environ["AZUREML_BI_OUTPUT_PATH"]
    output_file = os.path.join(output_path, f"{task_name}.jsonl")


def run(mini_batch: List[str]):
    if mini_batch:
        filtered_files = filter(lambda x: Path(x).suffix in DATA_READERS, mini_batch)
        results = []

        for file in filtered_files:
            data_format = Path(file).suffix
            data = load_dataset(DATA_READERS[data_format], data_files={"data": file})[
                "data"
            ].data.to_pandas()
            if text_column:
                data = data.loc[[text_column]]
            scores = model.predict(data)
            results.append(
                pd.DataFrame(
                    {
                        "file": np.repeat(Path(file).name, len(scores)),
                        "row": range(0, len(scores)),
                        task_name: scores,
                    }
                )
            )

        pd.concat(results, axis="rows").to_json(
            output_file, orient="records", mode="a", lines=True
        )

    return mini_batch
