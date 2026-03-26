from azure.identity import ManagedIdentityCredential
from azure.keyvault.secrets import SecretClient
import json
import logging
import os
import time

logger = logging.getLogger("keyvault-score")
multiplier: int | None = None


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

    for env_name, env_value in os.environ.items():
        if "KV_SECRET" in env_name:
            try:
                secret_name, vault_url = env_value.split("@", maxsplit=1)
            except ValueError:
                raise ValueError(
                    f"Wrong value format for env var {env_name} with value {env_value}. Should be of the form <SECRET_NAME>@<VAULT_URL>"
                )

            if vault_url in secret_clients:
                secret_client = secret_clients[vault_url]
            else:
                secret_client = SecretClient(vault_url=vault_url, credential=credential)
                secret_clients[vault_url] = secret_client

            secret_value = secret_client.get_secret(secret_name).value
            os.environ[env_name] = secret_value


def resolve_multiplier(retries=1, delay_seconds=5):
    global multiplier

    if multiplier is not None:
        return multiplier

    last_error = None

    for attempt in range(retries):
        try:
            load_secrets()
            multiplier = int(os.getenv("KV_SECRET_MULTIPLIER"))
            return multiplier
        except Exception as exc:
            last_error = exc
            if attempt < retries - 1:
                time.sleep(delay_seconds)

    raise RuntimeError("Failed to load multiplier secret from Key Vault.") from last_error


def init():
    try:
        resolve_multiplier()
    except Exception as exc:
        logger.warning(
            "Key Vault secret is not ready during container startup. Requests will retry secret resolution. Error: %s",
            exc,
        )


def run(data):
    current_multiplier = resolve_multiplier(retries=12, delay_seconds=5)
    data = json.loads(data)
    model_input = data["input"]
    output = model_input * current_multiplier

    return {"output": output}
