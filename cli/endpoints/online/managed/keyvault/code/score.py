from azure.identity import ManagedIdentityCredential
from azure.keyvault.secrets import SecretClient
import os
import json

secret_client : SecretClient = None

def init():
    global secret_client
    cred = ManagedIdentityCredential()
    print(cred)
    print(os.environ)
    kv_name = os.getenv("KV_NAME") 
    secret_client = SecretClient(vault_url=f"https://{kv_name}.vault.azure.net", credential=cred)

def run(data): 

    try:
        data = json.loads(data)
        name = data["name"]
        secret = secret_client.get_secret(name=name)
        return {"secret" : secret.value}
    except Exception as e: 
        return repr(e)

