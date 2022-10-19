import argparse
import datetime
from pathlib import Path
import yaml
from mltable import load


def parse_args():
    # setup arg parser
    parser = argparse.ArgumentParser()

    # add arguments
    parser.add_argument("--train_data", type=str)
    parser.add_argument("--preprocess_train_data", type=str)
    # parse args
    args = parser.parse_args()
    print("args received ", args)
    # return args
    return args


def get_preprocessed_data(dataframe):
    """
    Do preprocessing as needed
    Currently we are just passing pandas dataframe as it is
    """
    return dataframe


def main(args):
    """
    Preprocessing of training/validation data
    """
    train_data_table = load(args.train_data)
    train_dataframe = train_data_table.to_pandas_dataframe()
    preprocessed_train_dataframe = get_preprocessed_data(train_dataframe)

    # write preprocessed train data in output path
    preprocessed_train_dataframe.to_csv(
        args.preprocess_train_data + "/nyc_energy_training_clean.csv",
        index=False,
        header=True,
    )

    # Write MLTable yaml file as well in output folder
    # Since in this example we are not doing any preprocessing, we are just copying same yaml file from input,change it if needed

    # read and write MLModel yaml file for train data
    with open(args.train_data + "/MLTable", "r") as file:
        yaml_file = yaml.safe_load(file)
    with open(args.preprocess_train_data + "/MLTable", "w") as file:
        yaml.dump(yaml_file, file)


# run script
if __name__ == "__main__":
    # parse args
    args = parse_args()

    # run main function
    main(args)
