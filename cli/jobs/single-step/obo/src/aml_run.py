# imports
import os
import argparse
from azure.ai.ml.identity import AzureMLOnBehalfOfCredential
from azure.ai.ml import MLClient, Input, command


# define functions
def main(args):
    job_definition = get_job_definition(args)
    ml_client = get_ml_client(args)
    returned_job = ml_client.jobs.create_or_update(job_definition)
    print(returned_job)


def get_ml_client(args):
    credential = AzureMLOnBehalfOfCredential()
    client = MLClient(
        credential=credential,
        subscription_id=args.subscription,
        resource_group_name=args.resource_group,
        workspace_name=args.workspace_name,
    )
    return client

def get_job_definition(args):
    job = command(
        code="./src",  # local path where the code is stored
        command="python main.py --iris-csv ${{inputs.iris}} --C ${{inputs.C}} --kernel ${{inputs.kernel}} --coef0 ${{inputs.coef0}}",
        inputs={
            "iris": Input(
                type="uri_file",
                path="https://azuremlexamples.blob.core.windows.net/datasets/iris.csv"),
            "C": args.C,
            "kernel": args.kernel,
            "coef0": args.coef0,},
        environment="AzureML-sklearn-0.24-ubuntu18.04-py37-cpu:22",
        compute="cpu-cluster-1",
        display_name="sklearn-iris-example",
        description="sklearn iris example",
        experiment_name=args.experiment_name,
        tags = {"starter_run" : os.environ.get("MLFLOW_RUN_ID") },
        # environment_variables={
        #     "AZ_BATCHAI_CONFIG_EnableDefaultLogFilteringPatterns": "false"
        # }
    )
    
    return job


def parse_args():
    # setup arg parser
    parser = argparse.ArgumentParser()

    # add arguments
    parser.add_argument("--subscription", type=str, default='b17253fa-f327-42d6-9686-f3e553e24763')
    parser.add_argument("--resource_group", type=str, default='anksing_rg')
    parser.add_argument("--workspace_name", type=str, default='anksing_wcus')
    parser.add_argument("--experiment_name", type=str, default='obo-aml-example')
    parser.add_argument("--iris-csv", type=str)
    parser.add_argument("--C", type=float, default=1.0)
    parser.add_argument("--kernel", type=str, default="rbf")
    parser.add_argument("--degree", type=int, default=3)
    parser.add_argument("--gamma", type=str, default="scale")
    parser.add_argument("--coef0", type=float, default=0)
    parser.add_argument("--shrinking", type=bool, default=False)
    parser.add_argument("--probability", type=bool, default=False)
    parser.add_argument("--tol", type=float, default=1e-3)
    parser.add_argument("--cache_size", type=float, default=1024)
    parser.add_argument("--class_weight", type=dict, default=None)
    parser.add_argument("--verbose", type=bool, default=False)
    parser.add_argument("--max_iter", type=int, default=-1)
    parser.add_argument("--decision_function_shape", type=str, default="ovr")
    parser.add_argument("--break_ties", type=bool, default=False)
    parser.add_argument("--random_state", type=int, default=42)

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
