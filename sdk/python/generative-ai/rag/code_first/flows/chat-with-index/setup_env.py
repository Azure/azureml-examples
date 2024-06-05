import os
from typing import Union

from promptflow import tool
from promptflow.connections import AzureOpenAIConnection, OpenAIConnection


@tool
def setup_env(connection: Union[AzureOpenAIConnection, OpenAIConnection], config: dict):
    if not connection or not config:
        return

    if isinstance(connection, AzureOpenAIConnection):
        os.environ["OPENAI_API_TYPE"] = "azure"
        os.environ["OPENAI_API_BASE"] = connection.api_base
        os.environ["OPENAI_API_KEY"] = connection.api_key
        os.environ["OPENAI_API_VERSION"] = connection.api_version

    if isinstance(connection, OpenAIConnection):
        os.environ["OPENAI_API_KEY"] = connection.api_key
        if connection.organization is not None:
            os.environ["OPENAI_ORG_ID"] = connection.organization

    for key in config:
        os.environ[key] = str(config[key])

    return "Ready"
