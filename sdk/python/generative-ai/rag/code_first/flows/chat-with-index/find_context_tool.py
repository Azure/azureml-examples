from promptflow import tool
from src.find_context import find_context


@tool
def find_context_tool(question: str, mlindex_uri: str):
    prompt, documents = find_context(question, mlindex_uri)

    return {"prompt": prompt, "context": [d.page_content for d in documents]}
