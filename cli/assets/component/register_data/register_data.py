# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import argparse
from pathlib import Path
import pickle

import mlflow

from azureml.core import Run

# Get run
run = Run.get_context()
run_id = run.get_details()["runId"]
print(run_id)

# Default datastore
ws = run.experiment.workspace
print('ws', ws)
default_store = ws.get_default_datastore() 
print('ds', default_store)
def parse_args():

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, help='input path of data')
    parser.add_argument('--data_name', type=str, help='name of file dataset')
    
    args, _ = parser.parse_known_args()
    print(f'Arguments: {args}')

    return args


def main():

    args = parse_args()

    dir = args.data_path
    data_name = args.data_name

    # upload data to data store
    default_store.upload(dir, 
                        target_path = run_id, 
                        overwrite = True, 
                        show_progress = True)        

    print("upload data ", run_id)
    
    print("registing data set", data_name)
    from azureml.core import Dataset
    data_set = Dataset.File.from_files(default_store.path(run_id))
    data_set.register(ws, data_name)
    
    

if __name__ == "__main__":
    main()