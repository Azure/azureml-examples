# Chat with MLIndex

This is a simple flow that allow you to ask questions about the content of an MLIndex and get answers.
You can run the flow with a URL to an MLIndex and question as argument.
When you ask a question, it will look up the index to retrieve relevant content and post the question with the relevant content to OpenAI chat model (gpt-3.5-turbo or gpt4) to get an answer.

Tools used in this flow：
- custom `python` Tool

## Prerequisites

Install dependencies:
```bash
pip install -r requirements.txt
```

## Get started
### Create connection in this folder

```bash
# create connection needed by flow
if pf connection list | grep open_ai_connection; then
    echo "open_ai_connection already exists"
else
    pf connection create --file ./azure_openai.yml --name open_ai_connection --set api_key=<your_api_key> api_base=<your_api_base>
fi
```

### SDK Example

Refer to [local_docs_to_faiss_mlindex_with_promptfow.py](../../mlindex_local/local_docs_to_faiss_mlindex_with_promptfow.py)

### CLI Example

```bash
# test with flow inputs, you need local or remote MLIndex (refer to SDK examples to create them)
pf flow test --flow . --inputs question="" mlindex_uri="../../mlindex_local/mlindex_docs_aoai_faiss"

# (Optional) create a random run name
run_name="doc_questions_"$(openssl rand -hex 12)

# run with multiline data, --name is optional
pf run create --flow . --data ../data/rag_docs_questions.jsonl --stream --column-mapping question='${data.chat_input}' mlindex_uri='../../mlindex_local/mlindex_docs_aoai_faiss' chat_history='${data.chat_history}' config='{"CHAT_MODEL_DEPLOYMENT_NAME": "gpt-35-turbo", "PROMPT_TOKEN_LIMIT": "2000", "MAX_COMPLETION_TOKENS": "256", "VERBOSE": "True"}' --name $run_name

# visualize run output details
pf run visualize --name $run_name
```
