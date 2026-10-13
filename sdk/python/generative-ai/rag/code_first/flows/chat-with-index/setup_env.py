import os
from typing import Union

from promptflow import tool
from promptflow.connections import AzureOpenAIConnection, OpenAIConnection

ALLOWED_CONFIG_KEYS = {
    "CHAT_MODEL_DEPLOYMENT_NAME",
    "PROMPT_TOKEN_LIMIT",
    "MAX_COMPLETION_TOKENS",
    "VERBOSE",
}


@tool
def setup_env(connection: Union[AzureOpenAIConnection, OpenAIConnection], config: dict):
    if not connection or not config:
        return

    unsupported_keys = set(config) - ALLOWED_CONFIG_KEYS
    if unsupported_keys:
        raise ValueError(f"Unsupported configuration keys: {sorted(unsupported_keys)}")

    if isinstance(connection, AzureOpenAIConnection):
        os.environ["OPENAI_API_TYPE"] = "azure"
        os.environ["OPENAI_API_BASE"] = connection.api_base
        os.environ["OPENAI_API_KEY"] = connection.api_key
        os.environ["OPENAI_API_VERSION"] = connection.api_version

    if isinstance(connection, OpenAIConnection):
        os.environ["OPENAI_API_KEY"] = connection.api_key
        if connection.organization is not None:
            os.environ["OPENAI_ORG_ID"] = connection.organization

    for key, value in config.items():
        os.environ[key] = str(value)

    return "Ready"
