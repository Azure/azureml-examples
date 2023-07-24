from langchain.agents import load_tools
from langchain.tools import AIPluginTool
from parse import *
from langchain.chat_models.base import BaseChatModel
from langchain.chat_models import ChatOpenAI, AzureChatOpenAI

import utils


def create_plugins_static():
    plugins = [
        AIPluginTool.from_plugin_url(
            "https://www.klarna.com/.well-known/ai-plugin.json"
        )
    ]
    plugins += load_tools(["requests_all"])
    return plugins


def create_chat_model(openai_config: utils.OpenAIConfig) -> BaseChatModel:
    if openai_config.is_azure_openai():
        return AzureChatOpenAI(
            temperature=0,
            openai_api_base=openai_config.AZURE_OPENAI_API_ENDPOINT,
            openai_api_version=openai_config.AZURE_OPENAI_API_VERSION
            if openai_config.AZURE_OPENAI_API_VERSION
            else "2023-03-15-preview",
            deployment_name=openai_config.AZURE_OPENAI_API_DEPLOYMENT_NAME,
            openai_api_key=openai_config.OPENAI_API_KEY,
            openai_api_type=openai_config.OPENAI_API_TYPE,
        )
    else:
        return ChatOpenAI(
            temperature=0,
            openai_api_key=openai_config.OPENAI_API_KEY,
            openai_organization=openai_config.OPENAI_ORG_ID,
            model_name=openai_config.OPENAI_MODEL_ID,
        )
