import simple_agent_app
from langchain.tools import AIPluginTool
from langchain.agents import load_tools
import os, sys

sys.path.insert(0, str(os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))))

import utils

plugins = [
    AIPluginTool.from_plugin_url("https://www.klarna.com/.well-known/ai-plugin.json")
]
plugins += load_tools(["requests_all"])

agent = simple_agent_app.SimpleAgentApp(
    openai_config=utils.OpenAIConfig.from_env(), plugins=plugins
)

agent.run("what are the top 5 results for womens t shirts on klarna?")
