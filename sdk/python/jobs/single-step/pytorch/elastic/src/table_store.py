"""Class for storing and retrieving data from Azure Table Storage."""
import base64
import datetime

from azure.core.exceptions import ResourceNotFoundError
from azure.data.tables import TableClient, TableServiceClient
from torch.distributed import Store
from utils import get_table_name


# TODO: Use the same implementation as etcd_store.py
class AzureTableStore(Store):
    """
    This class implements a key-value store for PyTorch distributed training using Azure Table Storage.
    """

    job_id: str
    table_service_client: TableServiceClient
    table_name: str

    def __init__(self, job_id: str, table_service_client: TableServiceClient, timeout=datetime.timedelta(seconds=30)):
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
        self.table_client.upsert_entity({
            'PartitionKey': base64.b64encode(key.encode()).decode(),
            'RowKey': base64.b64encode(key.encode()).decode(),
            'Value': base64.b64encode(value).decode()
        })

    def get(self, key):
        print(f"Getting key: {key}")
        import time
        import datetime
        start = time.time()
        while datetime.timedelta(seconds=time.time() - start) < self._timeout:
            try:
                entity = self.table_client.get_entity(
                    base64.b64encode(key.encode()).decode(),
                    base64.b64encode(key.encode()).decode())
                return base64.b64decode(entity['Value'])
            except ResourceNotFoundError:
                print(f"Key {key} doesn't exist yet. Sleeping and trying again to read table...")
                time.sleep(5)
        raise LookupError(f'Key {key} not found in timeout {self._timeout}')

    def delete_key(self, key):
        print(f"Deleting key: {key}")
        pk = base64.b64encode(key.encode("utf-8")).decode("utf-8")
        rk = base64.b64encode(key.encode("utf-8")).decode("utf-8")
        self.table_client.delete_entity(pk, rk)

    def set_timeout(self, timeout):
        print(f"Setting timeout: {timeout}")
        self._timeout = timeout
