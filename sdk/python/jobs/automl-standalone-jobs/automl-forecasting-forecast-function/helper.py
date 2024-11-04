# Generate synthetic data

import pandas as pd
import numpy as np


def get_timeseries(
    train_len: int,
    test_len: int,
    time_column_name: str,
    target_column_name: str,
    time_series_id_column_name: str,
    time_series_number: int = 1,
    freq: str = "H",
):
    """
    Return the time series of designed length.

    :param train_len: The length of training data (one series).
    :type train_len: int
    :param test_len: The length of testing data (one series).
    :type test_len: int
    :param time_column_name: The desired name of a time column.
    :type time_column_name: str
    :param time_series_number: The number of time series in the data set.
    :type time_series_number: int
    :param freq: The frequency string representing pandas offset.
                 see https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html
    :type freq: str
    :returns: the tuple of train and test data sets.
    :rtype: tuple

    """
    data_train = []  # type: List[pd.DataFrame]
    data_test = []  # type: List[pd.DataFrame]
    data_length = train_len + test_len
    for i in range(time_series_number):
        X = pd.DataFrame(
            {
                time_column_name: pd.date_range(
                    start="2000-01-01", periods=data_length, freq=freq
                ),
                target_column_name: np.arange(data_length).astype(float)
                + np.random.rand(data_length)
                + i * 5,
                "ext_predictor": np.asarray(range(42, 42 + data_length)),
                time_series_id_column_name: np.repeat("ts{}".format(i), data_length),
            }
        )
        data_train.append(X[:train_len])
        data_test.append(X[train_len:])
    X_train = pd.concat(data_train)
    y_train = X_train.pop(target_column_name).values
    X_test = pd.concat(data_test)
    y_test = X_test.pop(target_column_name).values
    return X_train, y_train, X_test, y_test


def make_forecasting_query(
    fulldata, time_column_name, target_column_name, forecast_origin, horizon, lookback
):

    """
    This function will take the full dataset, and create the query
    to predict all values of the time series from the `forecast_origin`
    forward for the next `horizon` horizons. Context from previous
    `lookback` periods will be included.



    fulldata: pandas.DataFrame           a time series dataset. Needs to contain X and y.
    time_column_name: string             which column (must be in fulldata) is the time axis
    target_column_name: string           which column (must be in fulldata) is to be forecast
    forecast_origin: datetime type       the last time we (pretend to) have target values
    horizon: timedelta                   how far forward, in time units (not periods)
    lookback: timedelta                  how far back does the model look

    Example:


    ```

    forecast_origin = pd.to_datetime("2012-09-01") + pd.DateOffset(days=5) # forecast 5 days after end of training
    print(forecast_origin)

    X_query, y_query = make_forecasting_query(data,
                       forecast_origin = forecast_origin,
                       horizon = pd.DateOffset(days=7), # 7 days into the future
                       lookback = pd.DateOffset(days=1), # model has lag 1 period (day)
                      )

    ```
    """

    X_past = fulldata[
        (fulldata[time_column_name] > forecast_origin - lookback)
        & (fulldata[time_column_name] <= forecast_origin)
    ]

    X_future = fulldata[
        (fulldata[time_column_name] > forecast_origin)
        & (fulldata[time_column_name] <= forecast_origin + horizon)
    ]

    y_past = X_past.pop(target_column_name).values.astype(float)
    y_future = X_future.pop(target_column_name).values.astype(float)

    # Now take y_future and turn it into question marks
    y_query = y_future.copy().astype(float)  # because sometimes life hands you an int
    y_query.fill(np.nan)

    print("X_past is " + str(X_past.shape) + " - shaped")
    print("X_future is " + str(X_future.shape) + " - shaped")
    print("y_past is " + str(y_past.shape) + " - shaped")
    print("y_query is " + str(y_query.shape) + " - shaped")

    X_pred = pd.concat([X_past, X_future])
    y_pred = np.concatenate([y_past, y_query])
    return X_pred, y_pred
