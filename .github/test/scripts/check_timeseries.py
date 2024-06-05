# This is used to check the results for a historical time series experiment
# The parameters are:
# 	--experiment_name            The name of the experiment to check
# 	--file_name                  The name of the notebook output file
#       --folder                     The notebook folder

import sys
import argparse
import os

from azureml.core.experiment import Experiment
from azureml.core.workspace import Workspace
from azureml.train.automl.run import AutoMLRun
from azureml.core.run import Run

parser = argparse.ArgumentParser()
parser.add_argument("--experiment_name")
parser.add_argument("--file_name")
parser.add_argument("--folder")

inputArgs = parser.parse_args()


def get_hts_run_type(run):
    children = list(run.get_children())
    return children[1].type


def check_training_run(step_runs, pipeline_run):
    assert len(step_runs) == 6, "HTS training runs {} should have 6 steps.".format(
        pipeline_run.id
    )
    automl_parents = []
    for step in step_runs:
        if step.name == "hts-automl-training":
            print("Checking AutoML runs for pipeline {}".format(pipeline_run.id))
            automl_parents = list(step.get_children())
            for automl in automl_parents:
                assert (
                    automl.status == "Completed"
                ), "AutoML run {} should be in Completed state.".format(automl.id)
    assert (
        len(automl_parents) > 0
    ), "Run {} should have at least one automl run.".format(pipeline_run.id)


def check_hts_experiment(experiment_name, file_name, folder):
    ws = Workspace.from_config(folder)
    experiment = Experiment(ws, experiment_name)
    runs = list(experiment.get_runs())
    runs = getNotebookRuns(runs, file_name, folder)

    assert len(runs) > 0, "At least one pipelines needs to be triggered."
    n_hts_training = 0
    n_hts_inferencing = 0
    for r in runs:
        print("Checking pipeline run {}".format(r.id))
        assert r.status == "Completed", "Run {} should be in Completed state.".format(
            r.id
        )
        assert r.type == "azureml.PipelineRun", "Run {} should be pipeline run.".format(
            r.id
        )
        step_runs = list(r.get_children())
        print("Checking all steps status now for {}.".format(r.id))
        for s in step_runs:
            assert (
                s.status == "Completed"
            ), "Run {} of {} should be in Completed state.".format(s.id, r.id)
        if get_hts_run_type(r) == "azureml.HTSInferencing":
            print("Checking inferencing run.")
            n_hts_inferencing += 1
            assert (
                len(step_runs) == 3
            ), "Inferencing run {} should have 3 steps.".format(r.id)
        elif get_hts_run_type(r) == "azureml.HTSTraining":
            print("Checking training run.")
            n_hts_training += 1
            check_training_run(step_runs, r)


def getNotebookRuns(runs, file_name, folder):
    full_name = os.path.join(folder, file_name)
    notebook_runs = []

    with open(full_name, "r") as notebook_file:
        notebook_output = notebook_file.read()

        for run in runs:
            if run.id in notebook_output:
                notebook_runs.append(run)

    return notebook_runs


check_hts_experiment(inputArgs.experiment_name, inputArgs.file_name, inputArgs.folder)
