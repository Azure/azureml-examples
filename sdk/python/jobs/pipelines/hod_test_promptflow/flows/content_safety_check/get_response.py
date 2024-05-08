from promptflow import tool
from promptflow.connections import AzureOpenAIConnection
import requests
import bs4


@tool
def get_response(connection: AzureOpenAIConnection, request: str) -> str:    
    return request