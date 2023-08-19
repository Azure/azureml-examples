from typing import Any, Dict, Optional, List

import argparse
import json
import os
import re
import shutil

import pandas as pd

from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from azureml.core import ScriptRunConfig
from azureml.automl.core.shared import constants
from azureml.automl.core.shared.types import GrainType
from azureml.automl.runtime.shared.score import scoring

GRAIN = "time_series_id"
BACKTEST_ITER = "backtest_iteration"
ACTUALS = "actual_level"
PREDICTIONS = "predicted_level"
ALL_GRAINS = "all_sets"

FORECASTS_FILE = "forecast.csv"
SCORES_FILE = "scores.csv"
SCORES_FILE_GRAIN = "scores_per_grain.csv"
PLOTS_FILE = "plots_fcst_vs_actual.pdf"
PLOTS_FILE_GRAIN = "plots_fcst_vs_actual_per_grain.pdf"
RE_INVALID_SYMBOLS = re.compile("[: ]")


def _compute_metrics(
    df: pd.DataFrame, metrics: List[str], actual_col: str, fcst_col: str
):
    """
    Compute metrics for one data frame.

    :param df: The data frame which contains actual_level and predicted_level columns.
    :return: The data frame with two columns - metric_name and metric.
    """
    scores = scoring.score_regression(
        y_test=df[actual_col], y_pred=df[fcst_col], metrics=metrics
    )
    metrics_df = pd.DataFrame(list(scores.items()), columns=["metric_name", "metric"])
    metrics_df.sort_values(["metric_name"], inplace=True)
    metrics_df.reset_index(drop=True, inplace=True)
    return metrics_df


def _format_grain_name(grain: GrainType) -> str:
    """
    Convert grain name to string.

    :param grain: the grain name.
    :return: the string representation of the given grain.
    """
    if not isinstance(grain, tuple) and not isinstance(grain, list):
        return str(grain)
    grain = list(map(str, grain))
    return "|".join(grain)


def compute_all_metrics(
    fcst_df: pd.DataFrame,
    actual_col: str,
    fcst_col: str,
    ts_id_colnames: List[str],
    metric_names: Optional[List[set]] = None,
):
    """
    Calculate metrics per grain.

    :param fcst_df: forecast data frame. Must contain 2 columns: 'actual_level' and 'predicted_level'
    :param metric_names: (optional) the list of metric names to return
    :param ts_id_colnames: (optional) list of grain column names
    :return: dictionary of summary table for all tests and final decision on stationary vs nonstaionary
    """
    if not metric_names:
        metric_names = list(constants.Metric.SCALAR_REGRESSION_SET)

    if ts_id_colnames is None:
        ts_id_colnames = []

    metrics_list = []
    if ts_id_colnames:
        for grain, df in fcst_df.groupby(ts_id_colnames):
            one_grain_metrics_df = _compute_metrics(
                df, metric_names, actual_col, fcst_col
            )
            one_grain_metrics_df[GRAIN] = _format_grain_name(grain)
            metrics_list.append(one_grain_metrics_df)

    # overall metrics
    one_grain_metrics_df = _compute_metrics(fcst_df, metric_names, actual_col, fcst_col)
    one_grain_metrics_df[GRAIN] = ALL_GRAINS
    metrics_list.append(one_grain_metrics_df)

    # collect into a data frame
    return pd.concat(metrics_list)


def _draw_one_plot(
    df: pd.DataFrame,
    time_column_name: str,
    target_column_name: str,
    grain_column_names: List[str],
    columns_to_plot: List[str],
    pdf: PdfPages,
    plot_predictions=False,
) -> None:
    """
    Draw the single plot.

    :param df: The data frame with the data to build plot.
    :param time_column_name: The name of a time column.
    :param grain_column_names: The name of grain columns.
    :param pdf: The pdf backend used to render the plot.
    """
    if isinstance(grain_column_names, str):
        grain_column_names = [grain_column_names]
    fig, _ = plt.subplots(figsize=(20, 10))
    df = df.set_index(time_column_name)
    plt.plot(df[columns_to_plot])
    plt.xticks(rotation=45)
    if grain_column_names:
        grain_name = [df[grain].iloc[0] for grain in grain_column_names]
        plt.title(f"Time series ID: {_format_grain_name(grain_name)}")
    plt.legend(columns_to_plot)
    plt.close(fig)
    pdf.savefig(fig)


def calculate_scores_and_build_plots(
    input_dir: str, output_dir: str, automl_settings: Dict[str, Any]
):
    os.makedirs(output_dir, exist_ok=True)
    grains = automl_settings.get(constants.TimeSeries.GRAIN_COLUMN_NAMES)
    time_column_name = automl_settings.get(constants.TimeSeries.TIME_COLUMN_NAME)
    if grains is None:
        grains = []
    if isinstance(grains, str):
        grains = [grains]
    while BACKTEST_ITER in grains:
        grains.remove(BACKTEST_ITER)

    dfs = []
    for fle in os.listdir(input_dir):
        file_path = os.path.join(input_dir, fle)
        if os.path.isfile(file_path) and file_path.endswith(".csv"):
            df_iter = pd.read_csv(file_path, parse_dates=[time_column_name])
            for _, iteration in df_iter.groupby(BACKTEST_ITER):
                dfs.append(iteration)
    forecast_df = pd.concat(dfs, sort=False, ignore_index=True)
    # == Per grain-iteration analysis
    # To make sure plots are in order, sort the predictions by grain and iteration.
    ts_index = grains + [BACKTEST_ITER]
    forecast_df.sort_values(by=ts_index, inplace=True)
    pdf = PdfPages(os.path.join(output_dir, PLOTS_FILE))
    for _, one_forecast in forecast_df.groupby(ts_index):
        _draw_one_plot(one_forecast, time_column_name, grains, pdf)
    pdf.close()
    forecast_df.to_csv(os.path.join(output_dir, FORECASTS_FILE), index=False)
    metrics = compute_all_metrics(forecast_df, grains + [BACKTEST_ITER])
    metrics.to_csv(os.path.join(output_dir, SCORES_FILE), index=False)

    # == Per grain analysis
    pdf = PdfPages(os.path.join(output_dir, PLOTS_FILE_GRAIN))
    for _, one_forecast in forecast_df.groupby(grains):
        _draw_one_plot(one_forecast, time_column_name, grains, pdf)
    pdf.close()
    metrics = compute_all_metrics(forecast_df, grains)
    metrics.to_csv(os.path.join(output_dir, SCORES_FILE_GRAIN), index=False)


def run_remote_inference(
    test_experiment,
    compute_target,
    train_run,
    test_dataset,
    target_column_name,
    rolling_evaluation_step_size=1,
    inference_folder="./forecast",
):
    # Create local directory to copy the model.pkl and inference_script_naive.py files into.
    # These files will be uploaded to and executed on the compute instance.
    os.makedirs(inference_folder, exist_ok=True)
    shutil.copy("scripts/inference_script_naive.py", inference_folder)

    # Find the extension of the model file (.pkl or .pt)
    ls = train_run.get_file_names()  # list artifacts
    regex = re.compile("outputs/model[.](pt|pkl)")
    model_path = None
    for v in ls:
        matcher = regex.match(v)
        if matcher:
            model_path = matcher[0]
            break
    model_name = os.path.split(model_path)[-1]

    train_run.download_file(model_path, os.path.join(inference_folder, model_name))

    inference_env = train_run.get_environment()
    print("Finished getting training environment ...\n---")

    config = ScriptRunConfig(
        source_directory=inference_folder,
        script="inference_script_naive.py",
        arguments=[
            "--target_column_name",
            target_column_name,
            "--test_dataset",
            test_dataset.as_named_input(test_dataset.name),
            "--rolling_evaluation_step_size",
            rolling_evaluation_step_size,
        ],
        compute_target=compute_target,
        environment=inference_env,
    )

    print("Submitting experiment ...\n---")
    run = test_experiment.submit(
        config,
        tags={
            "training_run_id": train_run.id,
            "run_algorithm": train_run.properties["run_algorithm"],
            "valid_score": train_run.properties["score"],
            "primary_metric": train_run.properties["primary_metric"],
        },
    )

    run.log("run_algorithm", run.tags["run_algorithm"])
    return run
