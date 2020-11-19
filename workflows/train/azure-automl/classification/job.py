from azureml.core import Workspace, Experiment, Dataset
from azureml.train.automl import AutoMLConfig

ws = Workspace.from_config()

data = "https://automlsamplenotebookdata.blob.core.windows.net/automl-sample-notebook-data/creditcard.csv"
experiment_name = "fastai-mnist-example"
compute_name = "cpu-cluster"

dataset = Dataset.Tabular.from_delimited_files(data)
training_data, validation_data = dataset.random_split(percentage=0.8, seed=223)
label_column_name = 'Class'

automl_settings = {
    "n_cross_validations": 3,
    "primary_metric": 'average_precision_score_weighted',
    "enable_early_stopping": True,
    "max_concurrent_iterations": 2, # This is a limit for testing purpose, please increase it as per cluster size
    "experiment_timeout_hours": 0.25, # This is a time limit for testing purposes, remove it for real use cases, this will drastically limit ablity to find the best model possible
    "verbosity": logging.INFO,
}

automl_config = AutoMLConfig(task = 'classification',
                             debug_log = 'automl_errors.log',
                             compute_target = compute_name,
                             training_data = training_data,
                             label_column_name = label_column_name,
                             **automl_settings
                            )

remote_run = Experiment(ws,experiment_name).submit(automl_config)
