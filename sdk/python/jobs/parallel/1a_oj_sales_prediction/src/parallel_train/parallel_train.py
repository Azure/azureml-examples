# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license.

import os
import glob
import pandas as pd
import numpy as np
import pickle
import mltable
import argparse
# from azureml.opendatasets import OjSalesSimulated
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingRegressor

def mape(y_true, y_pred):
    return round(np.mean(np.abs((y_true - y_pred) / y_true)) * 100, 2)

def init():
    
    parser = argparse.ArgumentParser()

    parser.add_argument("--drop_cols", type=str)
    parser.add_argument("--target_col", type=str)
    parser.add_argument("--date_col", type=str)
    parser.add_argument("--lagging_orders", type=str)
    parser.add_argument("--model_folder", type=str)
    parser.add_argument("--partition_keys", type=str)

    args, _ = parser.parse_known_args()

    global target_col
    target_col = args.target_col

    global date_col
    date_col = args.date_col

    global lagging_orders
    lagging_orders = [int(i) for i in args.lagging_orders.split(',')]

    global model_folder
    model_folder = args.model_folder
    
    global drop_cols 
    drop_cols = args.drop_cols.split(',')


def run(input_data, mini_batch_context):
    if(not isinstance(input_data, pd.DataFrame)):
        raise Exception("Not a valid DataFrame input.")
        
    if target_col not in input_data.columns:
        raise Exception("No target column found from input tabular data")
    elif date_col not in input_data.columns:
        raise Exception("No date column found from input tabular data")

    print(f"partition_key_value = {mini_batch_context.partition_key_value}")

    # data cleaning
    input_data[date_col] = pd.to_datetime(input_data[date_col])
    input_data = input_data.set_index(date_col).sort_index(ascending=True)
    input_data = input_data.assign(Week_Year=input_data.index.isocalendar().week.values)   
    input_data = input_data.drop(columns=drop_cols, errors='ignore')

    # data lagging
    max_lag_order = max(lagging_orders)
    train_tail = input_data.iloc[-max_lag_order:]
    column_order = train_tail.columns
    data_trans = input_data.copy()

    ## Make the lag features
    for lag_order in lagging_orders:
        data_trans['lag_' + str(lag_order)] = data_trans[target_col].shift(lag_order)

    data_trans = data_trans.loc[input_data.index]
    data_trans = data_trans.dropna()

    # traning & evaluation
    features = data_trans.columns.drop(target_col)

    X = data_trans[features].values
    y = data_trans[target_col].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05, random_state=12, shuffle=False)

    reg = GradientBoostingRegressor(random_state=12)
    reg.fit(X_train, y_train)
    reg_pred = reg.predict(X_test)

    reg_mape = mape(y_test, reg_pred)

    # Dump model 
    model_file_name = "my_model.pkl"
    relative_path = os.path.join(model_folder, *list(str(i) for i in mini_batch_context.partition_key_value.values()))
    print("##############################################")
    print("relative_path:")
    print(relative_path)
    print("##############################################")
    
    if not os.path.exists(relative_path):
        os.makedirs(relative_path)

    with open(os.path.join(relative_path, model_file_name), "wb") as f:
        pickle.dump(reg, f)
    
    # output perf 
    # returns from every mini-batch will be aggregated into 'append_row_to' output.
    result_df = pd.DataFrame()
    for key in mini_batch_context.partition_key_value.keys():
        result_df[key] = [mini_batch_context.partition_key_value[key]]
    
    result_df["MAPE"] = reg_mape

    return result_df
