"""
This method returns a reference to an existing datastore or create a new one
"""
from logging import getLogger
from msrest.exceptions import HttpOperationError
from azureml.core import Workspace
from azureml.core.datastore import Datastore

log = getLogger(__name__)


def get_blob_datastore(
    workspace: Workspace,
    get_default: bool,
    data_store_name: str,
    storage_name: str,
    storage_key: str,
    container_name: str,
):
    """
    Returns a reference to a datastore
    Parameters:
      workspace: str: existing AzureML Workspace object
      get_default: bool: get default datastore for the workspace
      data_store_name: str: data store name
      storage_name: str: blob storage account name
      storage_key: str: blob storage account key
      container_name: str: container name
    Returns:
      Datastore: a reference to datastore
    """
    blob_datastore = workspace.get_default_datastore()
    if get_default and blob_datastore:
        log.info("Using default datastore for the workspace")
    else:
        try:
            blob_datastore = Datastore.get(workspace, data_store_name)
            log.info("Found Blob Datastore with name: %s", data_store_name)
        except HttpOperationError:
            blob_datastore = Datastore.register_azure_blob_container(
                workspace=workspace,
                datastore_name=data_store_name,
                account_name=storage_name,  # Storage account name
                container_name=container_name,  # Name of Azure blob container
                account_key=storage_key,  # Storage account key
            )
        log.info("Registered blob datastore with name: %s", data_store_name)

    return blob_datastore
