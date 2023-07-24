from azure.keyvault.secrets import SecretClient
from azure.core.credentials import TokenCredential
import os

SECRET_VAULT_REF_KEY_START = "keyvaultref:"
secret_clients = {}


def load_secret(v, credential: TokenCredential):
    """
    secret_url: keyvaultref:https://mykeyvault.vault.azure.net/secrets/foo
    """

    try:
        start, secret_url = v.split(SECRET_VAULT_REF_KEY_START)

        vault_url, secret_name = secret_url.split("/secrets/")
    except ValueError:
        raise ValueError(
            f"Wrong value format for value {v}. Should be of the form keyvaultref:https://mykeyvault.vault.azure.net/secrets/foo"
        )

    if vault_url in secret_clients:
        secret_client = secret_clients[vault_url]
    else:
        secret_client = SecretClient(vault_url=vault_url, credential=credential)
        secret_clients[vault_url] = secret_client

    secret_value = secret_client.get_secret(secret_name).value
    return secret_value


def load_secrets(credential: TokenCredential):
    """
    Replaces the values of environment variables with names containing "KV_SECRET" and values of the form "<SECRET_NAME>@<VAULT_URL>" with the actual secret values

    Uses the ManagedIdentityCredential to create a SecretClient for each <VAULT_URL>. The endpoint's Managed Identity should have the get permission for secrets.

    Example:
        FOO: keyvaultref:https://mykeyvault.vault.azure.net/secrets/foo

        Will be replaced with the actual vaule of the secret named foo in mykeyvault.
    """
    for k, v in os.environ.items():
        if v.lower().startswith(SECRET_VAULT_REF_KEY_START):
            secret = load_secret(v, credential)
            print(f"Loaded secret for {k}, {secret[0:3]}*********")
            os.environ[k] = secret


class OpenAIConfig:
    OPENAI_API_TYPE: str = None
    OPENAI_API_KEY = None

    # required for OpenAI API
    OPENAI_ORG_ID = None
    OPENAI_MODEL_ID = "gpt-3.5-turbo"

    # required for Azure OpenAI API
    AZURE_OPENAI_API_ENDPOINT = None
    AZURE_OPENAI_API_DEPLOYMENT_NAME = None

    AZURE_OPENAI_API_VERSION = None

    @staticmethod
    def from_env():
        config = OpenAIConfig()
        for att in dir(config):
            if not att.startswith("__") and not callable(getattr(config, att)):
                config.__setattr__(
                    att, os.environ.get(att, config.__getattribute__(att))
                )
        return config

    def is_azure_openai(self):
        return self.OPENAI_API_TYPE and self.OPENAI_API_TYPE.lower() == "azure"
