# This is used to check the results for an experiment
# The parameters are:
# 	--experiment_name            The name of the experiment to check
# 	--file_name                  The name of the notebook output file
#       --folder                     The notebook folder
#       --metric_name                The name of the metric to check
#       --expected_num_iteration     The expected number of iterations.
#       --minimum_median_score       The minimum expected median score.
#       --absolute_minimum_score     The absolute minimum expected score.
#       --maximum_median_score       The maximum expected median score.
#       --absolute_maximum_score     The absolute maximum expected score.
#       --expected_run_count         The expected number of runs.
#       --vision_train_run           Indicates that this is a vission run.
#       --check_explanation_best_run Check the explanation of the best run.
#       --is_local_run               Indicates that this is a local run.

import argparse
import mlflow
import os
from mlflow.entities import ViewType
from mlflow.tracking.client import MlflowClient
from azure.identity import AzureCliCredential
from azure.ai.ml import automl, Input, MLClient, command

parser = argparse.ArgumentParser()
parser.add_argument("--experiment_name")
parser.add_argument("--file_name")
parser.add_argument("--folder")
parser.add_argument("--metric_name")
parser.add_argument("--expected_num_iteration", type=int)
parser.add_argument("--minimum_median_score", type=float)
parser.add_argument("--absolute_minimum_score", type=float)
parser.add_argument("--maximum_median_score", type=float)
parser.add_argument("--absolute_maximum_score", type=float)
parser.add_argument("--expected_run_count", type=int)
parser.add_argument("--vision_train_run", type=bool)
parser.add_argument("--check_explanation_best_run", type=bool)
parser.add_argument("--is_local_run", type=bool)

inputArgs = parser.parse_args()

def checkExperimentResult(
    experiment_name,
    file_name,
    folder,
    metric_name=None,
    expected_num_iteration=None,
    minimum_median_score=None,
    absolute_minimum_score=0.0,
    maximum_median_score=1.0,
    absolute_maximum_score=1.0,
    expected_run_count=1,
    vision_train_run=False,
):
    credential = AzureCliCredential()
    ml_client = MLClient.from_config(credential)

    MLFLOW_TRACKING_URI = ml_client.workspaces.get(
        name=ml_client.workspace_name
    ).mlflow_tracking_uri

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

    mlflow_client = MlflowClient()

    experiment = mlflow_client.get_experiment_by_name(experiment_name)

    print("Experimentid = " + experiment.experiment_id)

    runs = mlflow_client.search_runs(
        experiment_ids=experiment.experiment_id,
        filter_string='',
        run_view_type=ViewType.ALL,
        order_by=['run.info.start_time DESC'])

    print("Total runs: " + str(len(runs)))

    root_run_ids = getNotebookRuns(runs, file_name, folder)

    print("root_run_ids: "+str(root_run_ids))


    if vision_train_run:
        # Only check the most recent runs
        error_msg = (
            "Not enough runs found in " + ws.name + " for experiment " + experiment_name
        )
        assert len(root_run_ids) >= expected_run_count, error_msg
        runs = runs[:expected_run_count]
    print("Run count: {}".format(len(root_run_ids)))
    assert len(root_run_ids) == expected_run_count

    for root_run_id in root_run_ids:
        print("Validating run: " + root_run_id)
        children = getChildRuns(runs, root_run_id)

        if not vision_train_run:
            badScoreCount = 0
            goodScoreCount = 0
            # run_metrics = ml_run.get_metrics(recursive=True)

            for iteration in children:
                iteration_status = iteration.info.status
                print(iteration.info.run_id + ": " + iteration_status)
                assert iteration_status == "FINISHED" or iteration_status == "CANCELED"
                if iteration_status == "FINISHED":
                    metrics = iteration.data.metrics
                    print(metric_name + " = " + str(metrics[metric_name]))
                    assert metrics[metric_name] >= absolute_minimum_score
                    assert metrics[metric_name] <= absolute_maximum_score
                    if (
                        metrics[metric_name] < minimum_median_score
                        or metrics[metric_name] > maximum_median_score
                    ):
                        badScoreCount += 1
                    else:
                        goodScoreCount += 1
            assert badScoreCount < goodScoreCount
    print("check_experiment_result complete")



def getNotebookRuns(runs, file_name, folder):
    root_run_ids = set(run.data.tags['mlflow.rootRunId'] for run in runs if run.data.tags['mlflow.rootRunId']+"_setup"==run.info.run_id)
    full_name = os.path.join(folder, file_name)
    notebook_run_ids = []

    with open(full_name, "r") as notebook_file:
        notebook_output = notebook_file.read()

        return [runid for runid in root_run_ids if runid in notebook_output]

def getChildRuns(runs, root_run_id):
    return [run for run in runs if run.data.tags['mlflow.rootRunId']==root_run_id and run.info.run_id.replace(run.data.tags['mlflow.rootRunId']+"_", "").isdigit()]


checkExperimentResult(
    inputArgs.experiment_name,
    inputArgs.file_name,
    inputArgs.folder,
    inputArgs.metric_name,
    inputArgs.expected_num_iteration or 1000,
    inputArgs.minimum_median_score,
    inputArgs.absolute_minimum_score or 0.0,
    inputArgs.maximum_median_score or 1.0,
    inputArgs.absolute_maximum_score or 1.0,
    inputArgs.expected_run_count or 1,
    inputArgs.vision_train_run,
)

