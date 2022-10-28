import pandas as pd
import numpy as np

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def APE(actual, pred):
    """
    Calculate absolute percentage error.
    Returns a vector of APE values with same length as actual/pred.
    """
    return 100 * np.abs((actual - pred) / actual)


def MAPE(actual, pred):
    """
    Calculate mean absolute percentage error.
    Remove NA and values where actual is close to zero
    """
    not_na = ~(np.isnan(actual) | np.isnan(pred))
    not_zero = ~np.isclose(actual, 0.0)
    actual_safe = actual[not_na & not_zero]
    pred_safe = pred[not_na & not_zero]
    return np.mean(APE(actual_safe, pred_safe))


def calculate_metrics(actual, pred):
    not_na = ~(np.isnan(actual) | np.isnan(pred))
    actual_safe = actual[not_na]
    pred_safe = pred[not_na]
    rmse = np.sqrt(mean_squared_error(actual_safe, pred_safe))
    metrics_dict = {}
    metrics_dict["R2 score"] = r2_score(actual_safe, pred_safe)
    metrics_dict["mean absolute error"] = mean_absolute_error(actual_safe, pred_safe)
    metrics_dict["mean_absolute_percentage_error"] = MAPE(actual_safe, pred_safe)
    metrics_dict["root mean squared error"] = rmse
    
    return pd.DataFrame(metrics_dict.items(), columns=["metric name", "score"])
