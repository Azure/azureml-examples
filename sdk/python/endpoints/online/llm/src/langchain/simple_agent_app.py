from langchain.agents import initialize_agent, AgentType
import os, sys

# add parent directory to path
sys.path.insert(0, str(os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))))

import utils, langchain_utils


class SimpleAgentApp:
    openai_config: utils.OpenAIConfig = None
    _chain = None

    def __init__(self, openai_config: utils.OpenAIConfig = None, plugins=None):
        self.openai_config = openai_config
        llm = langchain_utils.create_chat_model(openai_config)
        self._chain = initialize_agent(
            plugins, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True
        )

    def run(self, question: str):
        return self._chain.run(question)
