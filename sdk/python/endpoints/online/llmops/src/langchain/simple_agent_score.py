import json
import os
import sys
from simple_agent_app import SimpleAgentApp
from azure.identity import DefaultAzureCredential
from parse import *

# add parent directory to path
sys.path.insert(0, str(os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))))
import langchain_utils
import utils

credential = DefaultAzureCredential(additionally_allowed_tenants=["*"])

OPENAI_API_KEY = None

"""
Required AzureML deployment score functions
    - init() - called during deployment creation
    - run() - called when invoking the endpoint deployment
"""


def init():
    utils.load_secrets(credential)

    # Load plugins into tools
    plugins = langchain_utils.create_plugins_static()

    global agent
    agent = SimpleAgentApp(openai_config=utils.OpenAIConfig.from_env(), plugins=plugins)


def run(raw_data):
    print(f"raw_data: {raw_data}")
    question = json.loads(raw_data)["question"]
    result = agent.run(question)
    return result
