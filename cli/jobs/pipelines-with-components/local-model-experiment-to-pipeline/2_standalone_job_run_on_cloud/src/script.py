#!/usr/bin/env python
# coding: utf-8

# Copyright (c) Microsoft Corporation. All rights reserved.

# # Predict NYC taxi fares by GBDT


from pathlib import Path
import os
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
import pickle
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
# import matplotlib.pyplot as plt
import mlflow
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--raw_data", type=str, help="Path to raw data")
parser.add_argument("--model_output", type=str, help="Path of output model")


args = parser.parse_args()

lines = [
    f"Raw data path: {args.raw_data}",
    f"model output path: {args.model_output}",

]

for line in lines:
    print(line)

# Read raw data from csv to dataframe
# raw_data = './../data/sample_data/'
print("raw data files: ")
arr = os.listdir(args.raw_data)
print(arr)

green_data = pd.read_csv((Path(args.raw_data) / 'greenTaxiData.csv'))
yellow_data = pd.read_csv((Path(args.raw_data) / 'yellowTaxiData.csv'))

print(green_data.shape)
print(yellow_data.shape)

# Define useful columns 
useful_columns = [
        "cost",
        "distance",
        "dropoff_datetime",
        "dropoff_latitude",
        "dropoff_longitude",
        "passengers",
        "pickup_datetime",
        "pickup_latitude",
        "pickup_longitude",
        "store_forward",
        "vendor",
    ]

print(useful_columns)

green_columns = {
        "vendorID": "vendor",
        "lpepPickupDatetime": "pickup_datetime",
        "lpepDropoffDatetime": "dropoff_datetime",
        "storeAndFwdFlag": "store_forward",
        "pickupLongitude": "pickup_longitude",
        "pickupLatitude": "pickup_latitude",
        "dropoffLongitude": "dropoff_longitude",
        "dropoffLatitude": "dropoff_latitude",
        "passengerCount": "passengers",
        "fareAmount": "cost",
        "tripDistance": "distance",
    }

yellow_columns = {
        "vendorID": "vendor",
        "tpepPickupDateTime": "pickup_datetime",
        "tpepDropoffDateTime": "dropoff_datetime",
        "storeAndFwdFlag": "store_forward",
        "startLon": "pickup_longitude",
        "startLat": "pickup_latitude",
        "endLon": "dropoff_longitude",
        "endLat": "dropoff_latitude",
        "passengerCount": "passengers",
        "fareAmount": "cost",
        "tripDistance": "distance",
    }


print("green_columns: ", green_columns)
print("yellow_columns: ", yellow_columns)

# Rename and remove columns
green_data.rename(columns=green_columns, inplace=True)
green_data_clean = green_data[useful_columns]
print('data size before removing columns: ', green_data.shape)
print('data size after removing columns: ', green_data_clean.shape)

yellow_data.rename(columns=yellow_columns, inplace=True)
yellow_data_clean = (yellow_data)[useful_columns]
print('data size before removing columns: ', yellow_data.shape)
print('data size after removing columns: ', yellow_data_clean.shape)

# Drop NaN
green_data_clean = green_data_clean.dropna(axis=0, how='any')
yellow_data_clea = yellow_data_clean.dropna(axis=0, how='any')

# Count total NaN at each column 
print(green_data_clean.isnull().sum())
print(yellow_data_clean.isnull().sum())

# Append yellow data to green data
combined_df = green_data_clean.append(yellow_data_clean, ignore_index=True)
combined_df.reset_index(inplace=True, drop=True)
print("combined data size:", combined_df.shape)

# Filter outliers by cost and disatnce.

combined_df_cost = combined_df[(combined_df.cost > 0) & (combined_df.cost  < 80)]
print('data size after removing cost outliers: %d' % len(combined_df_cost))

combined_df_cost_distance = combined_df_cost[(combined_df_cost.distance > 0) & (combined_df_cost.distance  < 25)]
print('data size after removing distance outliers: %d' % len(combined_df_cost_distance))

combined_df_cost_distance.reset_index(inplace=True, drop=True)

# Filter out coordinates for locations that are outside the city border.
combined_df_cost_distance = combined_df_cost_distance.astype(
    {
        "pickup_longitude": "float64",
        "pickup_latitude": "float64",
        "dropoff_longitude": "float64",
        "dropoff_latitude": "float64",
    }
)

latlong_filtered_df = combined_df_cost_distance[
    (combined_df_cost_distance.pickup_longitude <= -73.72)
    & (combined_df_cost_distance.pickup_longitude >= -74.09)
    & (combined_df_cost_distance.pickup_latitude <= 40.88)
    & (combined_df_cost_distance.pickup_latitude >= 40.53)
    & (combined_df_cost_distance.dropoff_longitude <= -73.72)
    & (combined_df_cost_distance.dropoff_longitude >= -74.72)
    & (combined_df_cost_distance.dropoff_latitude <= 40.88)
    & (combined_df_cost_distance.dropoff_latitude >= 40.53)
]

latlong_filtered_df.reset_index(inplace=True, drop=True)
print('data size after filtering locations:', latlong_filtered_df.shape)


# Split the pickup and dropoff date further into the day of the week, day of the month, and month values.
datetime_splited_df = pd.DataFrame(latlong_filtered_df)

temp = pd.DatetimeIndex(latlong_filtered_df["pickup_datetime"], dtype="datetime64[ns]")
datetime_splited_df["pickup_date"] = temp.date
datetime_splited_df["pickup_weekday"] = temp.dayofweek
datetime_splited_df["pickup_month"] = temp.month
datetime_splited_df["pickup_monthday"] = temp.day
datetime_splited_df["pickup_time"] = temp.time
datetime_splited_df["pickup_hour"] = temp.hour
datetime_splited_df["pickup_minute"] = temp.minute
datetime_splited_df["pickup_second"] = temp.second

temp = pd.DatetimeIndex(latlong_filtered_df["dropoff_datetime"], dtype="datetime64[ns]")
datetime_splited_df["dropoff_date"] = temp.date
datetime_splited_df["dropoff_weekday"] = temp.dayofweek
datetime_splited_df["dropoff_month"] = temp.month
datetime_splited_df["dropoff_monthday"] = temp.day
datetime_splited_df["dropoff_time"] = temp.time
datetime_splited_df["dropoff_hour"] = temp.hour
datetime_splited_df["dropoff_minute"] = temp.minute
datetime_splited_df["dropoff_second"] = temp.second

del datetime_splited_df["pickup_datetime"]
del datetime_splited_df["dropoff_datetime"]

datetime_splited_df.reset_index(inplace=True, drop=True)

# Drop the pickup_date, dropoff_date, pickup_time, dropoff_time columns because they're
# no longer needed (granular time features like hour,
# minute and second are more useful for model training).
del datetime_splited_df["pickup_date"]
del datetime_splited_df["dropoff_date"]
del datetime_splited_df["pickup_time"]
del datetime_splited_df["dropoff_time"]

print('data size after splitting datetime:',  datetime_splited_df.shape)

# Change the store_forward column to binary values
train_data = datetime_splited_df
train_data["store_forward"] = np.where((datetime_splited_df.store_forward == "N"), 0, 1)

# Split the data into input(X) and output(y)
y = train_data["cost"]
# X = train_data.drop(['cost'], axis=1)
X = train_data[
    [
        "distance",
        "dropoff_latitude",
        "dropoff_longitude",
        "passengers",
        "pickup_latitude",
        "pickup_longitude",
        "store_forward",
        "vendor",
        "pickup_weekday",
        "pickup_month",
        "pickup_monthday",
        "pickup_hour",
        "pickup_minute",
        "pickup_second",
        "dropoff_weekday",
        "dropoff_month",
        "dropoff_monthday",
        "dropoff_hour",
        "dropoff_minute",
        "dropoff_second",
    ]
]

# Split the data into train and test sets
trainX, testX, trainy, testy = train_test_split(X, y, test_size=0.3, random_state=42)
print(trainX.shape)
print(trainX.columns)

# Train a GBDT Regressor Model with the train set
learning_rate = 0.1
n_estimators = 100
model = GradientBoostingRegressor(learning_rate=learning_rate, n_estimators = n_estimators).fit(trainX, trainy)
print("training set score:", model.score(trainX, trainy))

# Output the model 
# model_output = './model/'
if not os.path.exists(args.model_output):
    os.mkdir(args.model_output)
pickle.dump(model, open((Path(args.model_output) / "model.sav"), "wb"))



# Make predictions on testX data and record them in a column named predicted_cost
predictions = model.predict(testX)

# Compare predictions to actuals (testy)
# The mean squared error
# print("Scored with the following model:\n{}".format(model))
# print("Mean squared error: %.2f" % mean_squared_error(testy, predictions))
# The coefficient of determination: 1 is perfect prediction
# print("Coefficient of determination: %.2f" % r2_score(testy, predictions))

# Log params and metrics to AML

mlflow.log_param("learning_rate", learning_rate)
mlflow.log_param("n_estimators", n_estimators)

mlflow.log_metric("mean_squared_error", mean_squared_error(testy, predictions))
mlflow.log_metric("r2_score", r2_score(testy, predictions))
