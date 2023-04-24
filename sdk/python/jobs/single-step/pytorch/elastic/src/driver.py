import subprocess

from torch.distributed.elastic.multiprocessing.errors import record

try:
    import azure.data.tables
except ImportError:
    print("azure-data-tables not found. Installing...")
    result = subprocess.run(["python", "-m", "pip", "install", "azure-data-tables"], capture_output=True, text=True)
    while result.returncode is None:
        pass
    # print(result.stdout)


import argparse
import os
import uuid

from torch.distributed.run import main as torchrun

from torch.distributed.elastic.rendezvous.api import RendezvousClosedError, rendezvous_handler_registry
from torch.distributed.elastic.rendezvous.dynamic_rendezvous import DynamicRendezvousHandler

from backend import AzureRendezvousBackend
from table_store import AzureTableStore
from azure.data.tables import TableServiceClient


if "DEBUG" in os.environ:
    is_local = True
    run_id = str(uuid.uuid4())
    print("Running locally")
else:
    is_local = False
    print("Running on Azure ML")
    from azureml.core import Run, Datastore

    run = Run.get_context()
    run_id = run.id
    ws = run.experiment.workspace
    datastore = Datastore.get_default(ws)


def get_table_service_client():
    if is_local:
        print("Creating Table Service Client by reading connection string from environment variable...")
        cs = os.environ["AZURE_STORAGE_CONNECTION_STRING"]
        table_service_client = TableServiceClient.from_connection_string(cs)
        return table_service_client

    def get_credentials(datastore: Datastore):
        from azure.core.credentials import AzureNamedKeyCredential
        from azure.core.credentials import AzureSasCredential

        if datastore.account_key is not None:
            return AzureNamedKeyCredential(datastore.account_name, datastore.account_key)
        if datastore.sas_token is not None:
            return AzureSasCredential(datastore.sas_token)
        # TODO: raise user error
        raise Exception(
            "Cannot find account key or SAS token in the specified datastore, "
            "and credential passthrough is not enabled. "
        )

    account_url = f"{datastore.protocol}://{datastore.account_name}.table.{datastore.endpoint}"
    credential = get_credentials(datastore)
    table_service_client = TableServiceClient(endpoint=account_url, credential=credential)
    return table_service_client


def create_azure_table_rendezvous_handler(params):
    print("rendezvous params")
    print(params.__dict__)

    table_service_client = get_table_service_client()
    store = AzureTableStore(params.run_id, table_service_client)
    backend = AzureRendezvousBackend(params.run_id, table_service_client)

    print(f"Creating backend with run id {params.run_id}, min nodes {params.min_nodes}, max nodes {params.max_nodes}")

    rdzv_handler = DynamicRendezvousHandler.from_backend(
        run_id=params.run_id, store=store, backend=backend, min_nodes=params.min_nodes, max_nodes=params.max_nodes
    )

    return rdzv_handler


def set_defaults(args):
    """Set default values for arguments that are not provided by the user."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--rdzv_backend", type=str)
    parser.add_argument("--rdzv_endpoint", type=str)
    parser.add_argument("--rdzv_id", type=str)
    parser.add_argument("--nnodes", type=str)

    known_args, unknown_args = parser.parse_known_args(args)

    if known_args.rdzv_backend is None:
        print("Setting rdzv_backend to 'azuretable'")
        known_args.rdzv_backend = "azuretable"

    if known_args.rdzv_id is None:
        print(f"Setting run id to {run_id}")
        known_args.rdzv_id = run_id

    if known_args.nnodes is None:
        nnodes = "1:50"
        print(f"Setting nnodes to {nnodes}")
        known_args.nnodes = nnodes

    known_args_list = [f"--{k}={v}" for k, v in vars(known_args).items()]
    return known_args_list + unknown_args


@record
def main(args=None):
    import torch.distributed.elastic.metrics as metrics

    # Handle RendezvousClosedError when the job is completed
    try:
        args = set_defaults(args)
        os.environ["LOGLEVEL"] = "INFO"
        os.environ["NCCL_DEBUG"] = "INFO"
        # os.environ["TORCH_DISTRIBUTED_DETAIL"] = "DEBUG"

        metrics.configure(metrics.ConsoleMetricHandler(), group="torchelastic")

        rendezvous_handler_registry.register("azuretable", create_azure_table_rendezvous_handler)
        print(f"Running torchrun with args {args}")
        torchrun(args)
    except RendezvousClosedError:
        print("Rendezvous closed. Exiting.")


if __name__ == "__main__":
    main()

    # main_args = [
    #     "--nnodes=1",
    #     "train.py",
    #     "--epochs=1",
    #     "--dist-backend=gloo",
    # ]
    # main(main_args)
