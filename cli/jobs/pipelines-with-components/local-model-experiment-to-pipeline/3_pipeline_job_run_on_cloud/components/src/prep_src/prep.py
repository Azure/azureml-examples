from pathlib import Path
import os
import pandas as pd
import argparse

parser = argparse.ArgumentParser("prep")
parser.add_argument("--raw_data", type=str, help="Path to raw data")
parser.add_argument("--prep_data", type=str, help="Path of prepped data")

args = parser.parse_args()

print("hello training world...")

lines = [
    f"Raw data path: {args.raw_data}",
    f"Data output path: {args.prep_data}",
]

for line in lines:
    print(line)

# read raw data from csv to dataframe
print("raw data files: ")
arr = os.listdir(args.raw_data)
print(arr)

green_data = pd.read_csv((Path(args.raw_data) / 'greenTaxiData.csv'))
yellow_data = pd.read_csv((Path(args.raw_data) / 'yellowTaxiData.csv'))



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

# print(useful_columns)

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


# print("green_columns: " + green_columns)
# print("yellow_columns: " + yellow_columns)


# rename and remove columns
green_data.rename(columns=green_columns, inplace=True)
green_data_clean = green_data[useful_columns]
print('data size before removing columns: ', green_data.shape)
print('data size after removing columns: ', green_data_clean.shape)

yellow_data.rename(columns=yellow_columns, inplace=True)
yellow_data_clean = (yellow_data)[useful_columns]
print('data size before removing columns: ', yellow_data.shape)
print('data size after removing columns: ', yellow_data_clean.shape)
 

# Append yellow data to green data
combined_df = green_data_clean.append(yellow_data_clean, ignore_index=True)
combined_df.reset_index(inplace=True, drop=True)
print("combined data size:", combined_df.shape)


combined_df.to_csv((Path(args.prep_data) / "prepped_data.csv"))