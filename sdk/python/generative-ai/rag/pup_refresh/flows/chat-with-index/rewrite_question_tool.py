from promptflow import tool
from src.rewrite_question import rewrite_question


@tool
def rewrite_question_tool(question: str, history: list, env_ready_signal: str):
    return rewrite_question(question, history)
