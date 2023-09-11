from jinja2 import Environment, FileSystemLoader
import os

from utils.oai import render_with_token_limit
from utils.logging import log

from azureml.rag.mlindex import MLIndex


def find_context(question: str, index_path: str):
    mlindex = MLIndex(index_path)
    index = mlindex.as_native_index_client()
    snippets = index.similarity_search(question, k=5)

    template = Environment(
        loader=FileSystemLoader(os.path.dirname(os.path.abspath(__file__)))
    ).get_template("qna_prompt.md")
    token_limit = int(os.environ.get("PROMPT_TOKEN_LIMIT"))

    # Try to render the template with token limit and reduce snippet count if it fails
    while True:
        try:
            prompt = render_with_token_limit(
                template, token_limit, question=question, context=enumerate(snippets)
            )
            break
        except ValueError:
            snippets = snippets[:-1]
            log(f"Reducing snippet count to {len(snippets)} to fit token limit")

    return prompt, snippets
