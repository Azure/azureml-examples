import os
os.system("pip install --upgrade azureml-train-automl")

from azureml.core import Workspace, Experiment, Dataset
from azureml.train.automl import AutoMLConfig

ws = Workspace.from_config()

data = "https://automlsamplenotebookdata.blob.core.windows.net/automl-sample-notebook-data/creditcard.csv"
experiment_name = "fastai-mnist-example"
compute_name = "cpu-cluster"

dataset = Dataset.Tabular.from_delimited_files(data)

automl_settings = {
    "n_cross_validations": 3,
    "primary_metric": "average_precision_score_weighted",
    "enable_early_stopping": True,
}

automl_config = AutoMLConfig(
    task="classification",
    debug_log="automl_errors.log",
    compute_target=compute_name,
    training_data=dataset,
    label_column_name="Class",
    **automl_settings
)

remote_run = Experiment(ws, experiment_name).submit(automl_config)
