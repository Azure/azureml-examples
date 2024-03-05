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
import os
from azureml.core.experiment import Experiment
from azureml.core.workspace import Workspace
from azureml.train.automl.run import AutoMLRun
from azureml.core.run import Run

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

try:
    from azureml.interpret import ExplanationClient
except ImportError:
    print(
        "azureml-interpret could not be imported for validation, not installed locally, skipping..."
    )


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
    ws = Workspace.from_config(folder)
    experiment = Experiment(ws, experiment_name)
    runs = list(experiment.get_runs(type="automl"))

    print("Total runs: " + str(len(runs)))

    runs = getNotebookRuns(runs, file_name, folder)

    if vision_train_run:
        # Only check the most recent runs
        error_msg = (
            "Not enough runs found in " + ws.name + " for experiment " + experiment_name
        )
        assert len(runs) >= expected_run_count, error_msg
        runs = runs[:expected_run_count]
    print("Run count: {}".format(len(runs)))
    assert len(runs) == expected_run_count

    for run in runs:
        print("Validating run: " + run.id)
        status = run.get_details()
        ml_run = AutoMLRun(experiment=experiment, run_id=run.id)
        children = list(ml_run.get_children())

        if vision_train_run:
            checkVisionTrainRun(children, minimum_median_score, maximum_median_score)
        else:
            properties = ml_run.get_properties()
            status = ml_run.get_details()
            print("Number of iterations found = " + properties["num_iterations"])
            assert properties["num_iterations"] == str(expected_num_iteration)
            badScoreCount = 0
            goodScoreCount = 0
            # run_metrics = ml_run.get_metrics(recursive=True)

            for iteration in children:
                iteration_status = iteration.status
                print(iteration.id + ": " + iteration_status)
                assert iteration_status == "Completed" or iteration_status == "Canceled"
                if iteration_status == "Completed":
                    props = iteration.get_properties()
                    if props.get("runTemplate") != "automl_child":
                        # not training iteration
                        continue
                    metrics = iteration.get_metrics()
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
        print("Run status: " + status["status"])
        assert status["status"] == "Completed"
    print("check_experiment_result complete")


def check_experiment_model_explanation_of_best_run(
    experiment_name, file_name, folder, is_local_run=False
):
    print("Start running check_experiment_model_explanation_of_best_run().")
    ws = Workspace.from_config(folder)

    experiment = Experiment(ws, experiment_name)
    automl_runs = list(experiment.get_runs(type="automl"))
    automl_runs = getNotebookRuns(automl_runs, file_name, folder)

    for run in automl_runs:
        print("Validating run: " + run.id)
        ml_run = AutoMLRun(experiment=experiment, run_id=run.id)

        if not is_local_run:
            model_explainability_run_id = ml_run.id + "_" + "ModelExplain"
            print("Checking the Model Explanation run: " + model_explainability_run_id)
            # Wait for the ME run to complete before accessing the result.
            model_explainability_run = Run(
                experiment=experiment, run_id=model_explainability_run_id
            )
            model_explainability_run.wait_for_completion()

        # The best run should have explanation result.
        best_run = ml_run.get_best_child()
        expl_client = ExplanationClient.from_run(best_run)

        # Download the engineered explanations
        engineered_explanations = expl_client.download_model_explanation(raw=False)
        assert engineered_explanations is not None
        importance_dict = engineered_explanations.get_feature_importance_dict()
        # Importance dict should not be empty.
        assert importance_dict is not None and importance_dict

        # Download the raw explanations
        raw_explanations = expl_client.download_model_explanation(raw=True)
        assert raw_explanations is not None
        importance_dict = raw_explanations.get_feature_importance_dict()
        # Importance dict should not be empty.
        assert importance_dict is not None and importance_dict

    print("check_experiment_model_explanation_of_best_run() completed.")


def checkVisionTrainRun(child_runs, expected_min_score, expected_max_score):
    for hd_run in child_runs:
        print(hd_run.id + ": " + hd_run.status)
        assert hd_run.status == "Completed"

        _, best_metric = hd_run._get_best_run_and_metric_value(
            include_failed=False, include_canceled=False
        )
        print("Primary metric value of {}: {}".format(hd_run.id, best_metric))

        lower_err_msg = (
            "Primary metric value was lower than the expected min value of {}".format(
                expected_min_score
            )
        )
        higher_err_msg = (
            "Primary metric value was higher than the expected max value of {}".format(
                expected_max_score
            )
        )
        assert best_metric >= expected_min_score, lower_err_msg
        assert best_metric <= expected_max_score, higher_err_msg


def checkVisionScoreRun(
    experiment_name,
    min_map_score=0.0,
    max_map_score=0.0,
    min_precision_score=0.0,
    max_precision_score=0.0,
    min_recall_score=0.0,
    max_recall_score=0.0,
    expected_run_count=1,
):
    ws = Workspace.from_config()
    experiment = Experiment(ws, experiment_name)
    runs = list(experiment.get_runs(type="azureml.scriptrun"))

    error_msg = (
        "Not enough runs found in " + ws.name + " for experiment " + experiment_name
    )
    assert len(runs) >= expected_run_count, error_msg
    runs = runs[:expected_run_count]
    print("azureml.scriptrun run type count: {}".format(len(runs)))
    assert len(runs) == expected_run_count

    for run in runs:
        print("Validating run: " + run.id)
        status = run.get_details()

        # Validation only implemented for object detection
        if experiment_name == "flickr47-logo-detection":
            metrics = run.get_metrics()
            checkMetric(
                metrics,
                run_id=run.id,
                metric_name="map",
                expected_min=min_map_score,
                expected_max=max_map_score,
            )
            checkMetric(
                metrics,
                run_id=run.id,
                metric_name="precision",
                expected_min=min_precision_score,
                expected_max=max_precision_score,
            )
            checkMetric(
                metrics,
                run_id=run.id,
                metric_name="recall",
                expected_min=min_recall_score,
                expected_max=max_recall_score,
            )

        print("Run status: " + status["status"])
        assert status["status"] == "Completed"
    print("checkVisionScoreRun complete")


def checkMetric(metrics, run_id, metric_name, expected_min, expected_max):
    score = metrics[metric_name]
    print("{} score of {}: {}".format(metric_name, run_id, score))

    lower_err_msg = "{} value was lower than the expected min value of {}".format(
        metric_name, expected_min
    )
    higher_err_msg = "{} value was higher than the expected max value of {}".format(
        metric_name, expected_max
    )
    assert score >= expected_min, lower_err_msg
    assert score <= expected_max, higher_err_msg


def getNotebookRuns(runs, file_name, folder):
    full_name = os.path.join(folder, file_name)
    notebook_runs = []

    with open(full_name, "r") as notebook_file:
        notebook_output = notebook_file.read()

        for run in runs:
            if run.id in notebook_output:
                notebook_runs.append(run)

    return notebook_runs


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

if inputArgs.check_explanation_best_run:
    check_experiment_model_explanation_of_best_run(
        inputArgs.experiment_name,
        inputArgs.file_name,
        inputArgs.folder,
        inputArgs.is_local_run,
    )
