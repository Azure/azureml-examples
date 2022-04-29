from pathlib import Path
import os
import pandas as pd
import numpy as np

import argparse

parser = argparse.ArgumentParser("transform")
parser.add_argument("--clean_data", type=str, help="Path to prepped data")
parser.add_argument("--transformed_data", type=str, help="Path of output data")

args = parser.parse_args()


lines = [
    f"Clean data path: {args.clean_data}",
    f"Transformed data output path: {args.transformed_data}",
]

for line in lines:
    print(line)


# read raw data from csv to dataframe
# raw_data = './data/'
print("clean data files: ")
arr = os.listdir(args.clean_data)
print(arr)

combined_df = pd.read_csv((Path(args.clean_data) / 'prepped_data.csv'))


# To eliminate incorrectly captured data points,
# filter filers by cost and disatnce.
# This step will significantly improve machine learning model accuracy,
# because data points with a zero cost or distance represent major outliers that throw off prediction accuracy.

combined_df_cost = combined_df[(combined_df.cost > 0) & (combined_df.cost  < 80)]
print('data size after removing cost filers: %d' % len(combined_df_cost))

combined_df_cost_distance = combined_df_cost[(combined_df_cost.distance > 0) & (combined_df_cost.distance  < 25)]
print('data size after removing distance filers: %d' % len(combined_df_cost_distance))

combined_df_cost_distance.reset_index(inplace=True, drop=True)

# Filter out coordinates for locations that are outside the city border.
# Chain the column filter commands within the filter() function
# and define the minimum and maximum bounds for each field

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
# To get the day of the week value, use the derive_column_by_example() function.
# The function takes an array parameter of example objects that define the input data,
# and the preferred output. The function automatically determines your preferred transformation.
# For the pickup and dropoff time columns, split the time into the hour, minute, and second by using
# the split_column_by_ xample() function with no example parameter. After you generate the new features,
# use the drop_columns() function to delete the original fields as the newly generated features are preferred.
# Rename the rest of the fields to use meaningful descriptions.

# datetime_splited_df =  latlong_filtered_df.copy()

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


# print(datetime_splited_df.dtypes)



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


# Output data
transformed_data = train_data.to_csv(
    (Path(args.transformed_data) / "transformed_data.csv")
)
