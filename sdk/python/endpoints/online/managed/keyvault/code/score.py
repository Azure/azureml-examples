from azure.identity import ManagedIdentityCredential
from azure.keyvault.secrets import SecretClient
import os
import json
import time
import logging

multiplier: int = None


def load_secrets():
    """
    Replaces the values of environment variables with names containing "KV_SECRET" and values of the form "<SECRET_NAME>@<VAULT_URL>" with the actual secret values

    Uses the ManagedIdentityCredential to create a SecretClient for each <VAULT_URL>. The endpoint's Managed Identity should have the get permission for secrets.

    Example:
        KV_SECRET_FOO: foo@https://keyvault123.vault.azure.net

        Will be replaced with the actual vaule of the secret named foo in keyvault123.
    """
    secret_clients = {}
    credential = ManagedIdentityCredential()

    for k, v in os.environ.items():
        if "KV_SECRET" in k:
            try:
                secret_name, vault_url = v.split("@")
            except ValueError:
                raise ValueError(
                    f"Wrong value format for env var {k} with value {v}. Should be of the form <SECRET_NAME>@<VAULT_URL>"
                )

            if vault_url in secret_clients:
                secret_client = secret_clients[vault_url]
            else:
                secret_client = SecretClient(vault_url=vault_url, credential=credential)
                secret_clients[vault_url] = secret_client

            # Retry to allow time for managed identity / access policy propagation
            for attempt in range(3):
                try:
                    secret_value = secret_client.get_secret(secret_name).value
                    break
                except Exception as e:
                    logging.warning(
                        f"Attempt {attempt + 1}/3 to get secret '{secret_name}' failed: {e}"
                    )
                    if attempt < 2:
                        time.sleep(10)
                    else:
                        raise
            os.environ[k] = secret_value


def init():
    load_secrets()

    global multiplier
    multiplier = int(os.getenv("KV_SECRET_MULTIPLIER"))


def run(data):
    data = json.loads(data)
    input = data["input"]
    output = input * multiplier

    return {"output": output}
