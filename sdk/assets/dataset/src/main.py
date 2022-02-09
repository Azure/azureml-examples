# imports
import mlflow
import argparse
import os

import pandas as pd

# define functions
def main(args):
    # enable auto logging
    mlflow.autolog()

    csv_file_name = 'titanic.csv' #here is the file we want to load
    file_found = False

    print('searching for file', csv_file_name)

    arr = os.listdir(args.input_dataset)
    for filename in arr:
        if filename == csv_file_name:
            print('found the file')
            file_found = True
            # read in data
            df = pd.read_csv(os.path.join(args.input_dataset, filename))
            index = df.index
            number_of_rows = len(index)
            print('Number of Rows:', number_of_rows)
            counter = 1
            for col in df.columns:
                print('Column',counter,':', col)
                counter = counter+1

    if file_found == False:
        print(csv_file_name, 'not found in the dataset')

def parse_args():
    # setup arg parser
    parser = argparse.ArgumentParser()

    # add arguments
    parser.add_argument("--input-dataset", type=str)

    # parse args
    args = parser.parse_args()

    # return args
    return args

# run script
if __name__ == "__main__":
    # parse args
    args = parse_args()

    # run main function
    main(args)
