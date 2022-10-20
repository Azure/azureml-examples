from azure.identity import ManagedIdentityCredential
from azure.keyvault.secrets import SecretClient
import os

secret_client = None 
secret = None
def init(): 
    global secret_client
    global secret

    kv_name = os.getenv("KV_NAME")
    secret_name = os.getenv("SECRET_NAME", "foo") 
    vault_url = f"https://{kv_name}.vault.azure.net/"
    credential = ManagedIdentityCredential()
    secret_client = SecretClient(vault_url=vault_url, credential=credential)   
    secret_client.get_secret(name=secret_name)

def run(data):
    return secret