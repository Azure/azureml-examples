import os
import argparse

from azure.identity import DefaultAzureCredential
from azure.ai.ml import MLClient

from azure.ai.ml.entities import Model
from azure.ai.ml.constants import ModelType

def parse_args():
    # setup arg parser
    parser = argparse.ArgumentParser()

    # add arguments
    parser.add_argument("--model_input_path", type=str, help="Path to input model")
    parser.add_argument(
        "--model_registeration_name", type=str, help="Name of the registered model"
    )

    # parse args
    args = parser.parse_args()
    print("Arguments received ",args)
    return args

def main(args):
    '''
    Register Model Example
    '''

    #Get Run ID from model path
    print("Getting model path")
    mlmodel_path = os.path.join(args.model_input_path, "MLmodel")
    runid = ""
    with open(mlmodel_path, "r") as modelfile:
        for line in modelfile:
            if "run_id" in line:
                runid = line.split(":")[1].strip()

    #Construct Model URI from run ID extract previously
    run_uri = "runs:/{}/outputs/".format(runid)

    # hardcoded as of now
    ml_client = MLClient(
    DefaultAzureCredential(), "********", "********", "**********")
    run_model = Model(
        path=run_uri,
        name=args.model_registeration_name,
        description="Model created from run.",
        type=ModelType.MLFLOW
    )

    ml_client.models.create_or_update(run_model)

if __name__ == "__main__":
    args = parse_args()
    main(args)