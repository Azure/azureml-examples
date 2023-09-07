import argparse
from dotenv import load_dotenv
import os

from qna import qna
from find_context import find_context
from rewrite_question import rewrite_question

# from build_index import create_faiss_index
# from utils.lock import acquire_lock


def chat_with_index(question: str, mlindex_uri: str, history: list):
    # with acquire_lock("create_folder.lock"):
    #     if not os.path.exists(".mlindex"):
    #         os.makedirs(".mlindex")

    # index_path = create_faiss_index(pdf_path)
    q = rewrite_question(question, history)
    prompt, context = find_context(q, mlindex_uri)
    stream = qna(prompt, history)

    return stream, context


def print_stream_and_return_full_answer(stream):
    answer = ""
    for str in stream:
        print(str, end="", flush=True)
        answer = answer + str + ""
    print(flush=True)

    return answer


def main_loop(mlindex_uri: str):
    load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))

    history = []
    while True:
        question = input("\033[92m" + "$User (type q! to quit): " + "\033[0m")
        if question == "q!":
            break

        stream, context = chat_with_index(question, mlindex_uri, history)

        print("\033[92m" + "$Bot: " + "\033[0m", end=" ", flush=True)
        answer = print_stream_and_return_full_answer(stream)
        history = history + [
            {"role": "user", "content": question},
            {"role": "assistant", "content": answer},
        ]


def main():
    parser = argparse.ArgumentParser(
        description="Ask questions about the contents of an MLIndex."
    )
    parser.add_argument("mlindex_uri", help="URI to MLIndex")
    args = parser.parse_args()

    main_loop(args.mlindex_uri)


if __name__ == "__main__":
    main()
