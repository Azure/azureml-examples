import pandas as pd

from azureml.metrics import constants
from azureml.metrics import compute_metrics


def calculate_metrics(
    X_train,
    X_test,
    actuals_colum_name,
    time_column_name,
    time_series_id_column_names=None,
    predictions_column_name="predicted",
):
    # Remove all NaNs in the train set
    X_train = X_train.copy()
    X_train.dropna(subset=[actuals_colum_name], inplace=True)
    y_train = X_train.pop(actuals_colum_name).values
    # Remove all NaNs in the test set.
    X_test = X_test.copy()
    X_test.dropna(subset=[actuals_colum_name, predictions_column_name], inplace=True)
    actual = X_test.pop(actuals_colum_name).values
    pred = X_test.pop(predictions_column_name).values
    metrics = compute_metrics(
        task_type=constants.Tasks.FORECASTING,
        y_test=actual,
        y_pred=pred,
        X_test=X_test,
        X_train=X_train,
        y_train=y_train,
        time_column_name=time_column_name,
        time_series_id_column_names=time_series_id_column_names,
        metrics=constants.Metric.SCALAR_REGRESSION_SET,
    )
    metrics_dict = metrics[constants.Metric.Metrics]
    return pd.DataFrame(metrics_dict.items(), columns=["metric name", "score"])
