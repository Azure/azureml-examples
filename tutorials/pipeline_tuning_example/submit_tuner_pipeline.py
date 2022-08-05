import argparse
from azure.ai.ml import dsl, Input, MLClient, load_component

# Authentication package
from azure.identity import DefaultAzureCredential, InteractiveBrowserCredential
# from azure.ai.ml.entities import Data
# from azure.ai.ml.constants import AssetTypes

import logging

from jinja2 import pass_environment

def remote_run():
    ################################################
    # connect to your Azure ML workspace
    ################################################
    try:
        credential = DefaultAzureCredential()
        # Check if given credential can get token successfully.
        credential.get_token("https://management.azure.com/.default")
    except Exception as ex:
        # Fall back to InteractiveBrowserCredential in case DefaultAzureCredential not work
        credential = InteractiveBrowserCredential()

    ml_client = MLClient(
        credential=credential,
        subscription_id=args.subscriptionId,
        resource_group_name=args.resourceGroup,
        workspace_name=args.workspace,
    )


    ################################################
    # load component functions
    ################################################
    pipeline_tuning_func = load_component(path="tuner/tuner_func.yaml")
    
    ################################################
    # build pipeline
    ################################################
    @dsl.pipeline(
        name="pipeline_tuning",
        default_compute_target="cpucluster",
    )
    def tuner_pipeline():
        tuner = pipeline_tuning_func()

    pipeline = tuner_pipeline()

    # submit the pipeline
    run = ml_client.jobs.create_or_update(
        pipeline,
        experiment_name="tuning AML pipeline",
    )

    # TODO: print the job url
    import webbrowser
    webbrowser.open(run.services["Studio"].endpoint)
    return run

def local_run():
    from tuner import tuner_func
    logger.info("Run tuner locally.")
    # tuner_func.tune_pipeline(concurrent_run=2)
    tuner_func.run_with_config({})


if __name__ == "__main__":
    # parser argument 
    parser = argparse.ArgumentParser()
    parser.add_mutually_exclusive_group(required=False)
    parser.add_argument(
        "--subscriptionId", type=str, default="48bbc269-ce89-4f6f-9a12-c6f91fcb772d"
    )
    parser.add_argument("--resourceGroup", type=str, default="aml1p-rg")
    parser.add_argument("--workspace", type=str, default="aml1p-ml-wus2")
    
    parser.add_argument('--remote', dest='remote', action='store_true')
    parser.add_argument('--local', dest='remote', action='store_false')
    # parser.set_defaults(remote=True)
    parser.set_defaults(remote=False)
    args = parser.parse_args()

    logger = logging.getLogger(__name__)

    if args.remote:
        remote_run()
    else:
        local_run()
