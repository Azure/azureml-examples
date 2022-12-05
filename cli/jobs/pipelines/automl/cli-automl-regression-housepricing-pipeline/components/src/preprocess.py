# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------

import os
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
    parser.add_argument("--validation_data", type=str)
    parser.add_argument("--test_data", type=str)
    parser.add_argument("--preprocessed_train_data", type=str)
    parser.add_argument("--preprocessed_validation_data", type=str)
    parser.add_argument("--preprocessed_test_data", type=str)
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
    Preprocessing of training/validation/test data
    """
    train_data_table = load(args.train_data)
    train_dataframe = train_data_table.to_pandas_dataframe()
    preprocessed_train_dataframe = get_preprocessed_data(train_dataframe)

    # write preprocessed train data in output path
    preprocessed_train_data_path = os.path.join(
        args.preprocessed_train_data, "house_pricing_train.csv"
    )
    preprocessed_train_dataframe.to_csv(
        preprocessed_train_data_path,
        index=False,
        header=True,
    )

    validation_data_table = load(args.validation_data)
    validation_dataframe = validation_data_table.to_pandas_dataframe()
    preprocessed_validation_dataframe = get_preprocessed_data(validation_dataframe)

    # write preprocessed validation data in output path
    preprocessed_validation_data_path = os.path.join(
        args.preprocessed_validation_data, "house_pricing_validation.csv"
    )
    preprocessed_validation_dataframe.to_csv(
        preprocessed_validation_data_path,
        index=False,
        header=True,
    )

    test_data_table = load(args.test_data)
    test_dataframe = test_data_table.to_pandas_dataframe()
    preprocessed_test_dataframe = get_preprocessed_data(test_dataframe)

    # write preprocessed validation data in output path
    preprocessed_test_data_path = os.path.join(
        args.preprocessed_test_data, "house_pricing_test.csv"
    )
    preprocessed_test_dataframe.to_csv(
        preprocessed_test_data_path, index=False, header=True
    )

    # Write MLTable yaml file as well in output folder
    # Since in this example we are not doing any preprocessing, we are just copying same yaml file from input,change it if needed

    # read and write MLModel yaml file for train data
    train_data_mltable_path = os.path.join(args.train_data, "MLTable")
    preprocessed_train_data_mltable_path = os.path.join(
        args.preprocessed_train_data, "MLTable"
    )
    with open(train_data_mltable_path, "r") as file:
        yaml_file = yaml.safe_load(file)
    with open(preprocessed_train_data_mltable_path, "w") as file:
        yaml.dump(yaml_file, file)

    # read and write MLModel yaml file for validation data
    validation_data_mltable_path = os.path.join(args.validation_data, "MLTable")
    preprocessed_validation_data_mltable_path = os.path.join(
        args.preprocessed_validation_data, "MLTable"
    )
    with open(validation_data_mltable_path, "r") as file:
        yaml_file = yaml.safe_load(file)
    with open(preprocessed_validation_data_mltable_path, "w") as file:
        yaml.dump(yaml_file, file)

    # read and write MLModel yaml file for validation data
    test_data_mltable_path = os.path.join(args.test_data, "MLTable")
    preprocessed_test_data_mltable_path = os.path.join(
        args.preprocessed_test_data, "MLTable"
    )
    with open(test_data_mltable_path, "r") as file:
        yaml_file = yaml.safe_load(file)
    with open(preprocessed_test_data_mltable_path, "w") as file:
        yaml.dump(yaml_file, file)


# run script
if __name__ == "__main__":
    # parse args
    args = parse_args()

    # run main function
    main(args)
