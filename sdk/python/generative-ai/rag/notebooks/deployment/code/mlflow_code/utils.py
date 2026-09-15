import os
from azureml.core import Workspace
from azureml.rag.utils.connections import get_connection_by_id_v2


def set_openai_env_vars(subscription_id, resource_group_name, workspace_name):
    ws = Workspace(
        subscription_id=subscription_id,
        resource_group=resource_group_name,
        workspace_name=workspace_name,
    )
    aoai_connection_id = os.environ["AZUREML_WORKSPACE_CONNECTION_ID_AOAI"]
    aoai_connection = get_connection_by_id_v2(aoai_connection_id)
    os.environ["OPENAI_API_TYPE"] = "azure"
    os.environ["OPENAI_API_VERSION"] = "2023-03-15-preview"
    os.environ["OPENAI_API_BASE"] = aoai_connection["properties"]["target"]
    os.environ["OPENAI_API_KEY"] = aoai_connection["properties"]["credentials"]["key"]
