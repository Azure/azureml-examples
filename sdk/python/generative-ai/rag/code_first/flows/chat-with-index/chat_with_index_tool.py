from promptflow import tool
from src.main import chat_with_index


@tool
def chat_with_index_tool(question: str, mlindex_uri: str, history: list, ready: str):
    history = convert_chat_history_to_chatml_messages(history)

    stream, context = chat_with_index(question, mlindex_uri, history)

    answer = ""
    for str in stream:
        answer = answer + str + ""

    return {"answer": answer, "context": context}


def convert_chat_history_to_chatml_messages(history):
    messages = []
    for item in history:
        messages.append({"role": "user", "content": item["inputs"]["question"]})
        messages.append({"role": "assistant", "content": item["outputs"]["answer"]})

    return messages


def convert_chatml_messages_to_chat_history(messages):
    history = []
    for i in range(0, len(messages), 2):
        history.append(
            {
                "inputs": {"question": messages[i]["content"]},
                "outputs": {"answer": messages[i + 1]["content"]},
            }
        )

    return history
