import subprocess
import time

from torch.distributed.elastic.multiprocessing.errors import record

try:
    import azure.data.tables
except ImportError:
    print("azure-data-tables not found. Installing...")
    result = subprocess.run(["python", "-m", "pip", "install", "azure-data-tables"], capture_output=True, text=True)
    while result.returncode is None:
        pass


import argparse
import base64
import binascii
import datetime
import os
import uuid
from typing import Optional, Tuple

from azure.core import MatchConditions
from azure.core.exceptions import ResourceNotFoundError
from azure.data.tables import TableClient, TableServiceClient
from torch.distributed import Store
from torch.distributed.elastic.rendezvous import RendezvousStateError
from torch.distributed.elastic.rendezvous.api import RendezvousClosedError, rendezvous_handler_registry
from torch.distributed.elastic.rendezvous.dynamic_rendezvous import DynamicRendezvousHandler, RendezvousBackend, Token
from torch.distributed.run import main as torchrun

if "DEBUG" in os.environ:
    is_local = True
    run_id = str(uuid.uuid4())
    print("DEV MODE: Running locally...")
else:
    is_local = False
    print("Running on Azure ML")
    from azureml.core import Datastore, Run

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
        from azure.core.credentials import AzureNamedKeyCredential, AzureSasCredential

        if datastore.account_key is not None:
            return AzureNamedKeyCredential(datastore.account_name, datastore.account_key)
        if datastore.sas_token is not None:
            return AzureSasCredential(datastore.sas_token)
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


def get_table_name(prefix, job_id):
    job_id = "".join([c for c in job_id if c.isalnum()]).lower()
    return f"{prefix}{job_id}"


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
        print(f"Setting TorchElastic Rendezvous run ID to {run_id}")
        known_args.rdzv_id = run_id

    if known_args.nnodes is None:
        nnodes = "1:50"
        print(f"Setting nnodes to {nnodes}")
        known_args.nnodes = nnodes

    known_args_list = [f"--{k}={v}" for k, v in vars(known_args).items()]
    return known_args_list + unknown_args


def enable_debugging():
    os.environ["NCCL_DEBUG"] = "INFO"
    os.environ["TORCH_CPP_LOG_LEVEL"] = "INFO"
    os.environ["TORCH_DISTRIBUTED_DEBUG"] = "DETAIL"
    os.environ["NCCL_DEBUG_SUBSYS"] = "COLL"
    os.environ["NCCL_DEBUG_FILE"] = "outputs/nccl_logs.txt"


class AzureTableStore(Store):
    """
    This class implements a key-value store for PyTorch distributed training using Azure Table Storage.
    """

    job_id: str
    table_service_client: TableServiceClient
    table_name: str

    def __init__(self, job_id: str, table_service_client: TableServiceClient, timeout=datetime.timedelta(seconds=300)):
        super(AzureTableStore, self).__init__()
        self.job_id = job_id
        self.table_service_client = table_service_client
        self.table_name = get_table_name("torch", self.job_id)
        if timeout is not None:
            self.set_timeout(timeout)
        print(f"Table name: {self.table_name}")
        self.table_client: TableClient = self.table_service_client.create_table_if_not_exists(self.table_name)

    def set(self, key, value):
        print(f"Setting key: {key}")
        self.table_client.upsert_entity(
            {
                "PartitionKey": base64.b64encode(key.encode()).decode(),
                "RowKey": base64.b64encode(key.encode()).decode(),
                "Value": base64.b64encode(value).decode(),
            }
        )

    def get(self, key):
        print(f"Getting key: {key}")
        start = time.time()
        while datetime.timedelta(seconds=time.time() - start) < self._timeout:
            try:
                entity = self.table_client.get_entity(
                    base64.b64encode(key.encode()).decode(), base64.b64encode(key.encode()).decode()
                )
                return base64.b64decode(entity["Value"])
            except ResourceNotFoundError:
                print(f"Key {key} doesn't exist yet. Sleeping and trying again to read table...")
                time.sleep(5)
        raise LookupError(f"Key {key} not found in timeout {self._timeout}")

    def delete_key(self, key):
        print(f"Deleting key: {key}")
        pk = base64.b64encode(key.encode("utf-8")).decode("utf-8")
        rk = base64.b64encode(key.encode("utf-8")).decode("utf-8")
        self.table_client.delete_entity(pk, rk)

    def set_timeout(self, timeout):
        print(f"Setting timeout: {timeout}")
        self._timeout = timeout


class AzureRendezvousBackend(RendezvousBackend):
    """Represents an Azure Table Storage-based rendezvous backend."""

    _prefix = "torchelastic"  # Prefix for the table name

    _run_id: str
    _table_service_client: TableServiceClient
    _table_client: TableClient
    _table_name: str
    _partition_key: str

    def __init__(
        self,
        run_id: str,
        table_service_client: TableServiceClient,
    ) -> None:
        self._run_id = run_id
        self._table_service_client = table_service_client
        self._table_name = get_table_name(self._prefix, run_id)
        self._table_client = self._table_service_client.create_table_if_not_exists(self._table_name)
        self._partition_key = "state"

    @property
    def name(self) -> str:
        """See base class."""
        return "azuretable"

    def get_state(self) -> Optional[Tuple[bytes, Token]]:
        """See base class."""
        try:
            entity = self._table_client.get_entity("state", "state")
            etag = entity.metadata["etag"]
            state = base64.b64decode(entity["State"])
            return state, etag
        except ResourceNotFoundError:
            return None

    def set_state(self, state: bytes, token: Optional[Token] = None) -> Optional[Tuple[bytes, Token, bool]]:
        entity = {"PartitionKey": "state", "RowKey": "state", "State": base64.b64encode(state).decode()}
        try:
            if token:
                upsert_result = self._table_client.update_entity(
                    entity, etag=token, match_condition=MatchConditions.IfNotModified
                )
            else:
                upsert_result = self._table_client.upsert_entity(entity)
            return state, upsert_result["etag"], True
        except Exception as e:
            # print(f"Exception writing to table:")
            # traceback.print_exc()
            pass
        curr_state = self.get_state()
        return curr_state[0], curr_state[1], False

    def _decode_state(self, entity: dict) -> Tuple[bytes, Token]:
        """Decodes the state from the entity and returns it along with the token."""
        base64_state = entity["State"]

        try:
            state = base64.b64decode(base64_state)
        except binascii.Error as exc:
            raise RendezvousStateError("The state object is corrupt. See inner exception for details.") from exc

        return state, entity.metadata["etag"]


@record
def main(args=None):
    try:
        args = set_defaults(args)

        # Uncomment to enable debug logs from NCCL / PyTorch / etc.
        # enable_debugging()

        # Register rendezvous handler for Azure Table Storage
        rendezvous_handler_registry.register("azuretable", create_azure_table_rendezvous_handler)

        print(f"Running torchrun with arguments: {args}")
        torchrun(args)
    except RendezvousClosedError:
        print("Rendezvous closed. Exiting.")


if __name__ == "__main__":
    main()
