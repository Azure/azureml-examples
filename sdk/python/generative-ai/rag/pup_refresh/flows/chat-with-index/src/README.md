# Chat with Index
This is a simple Python application that allow you to ask questions about the content of an MLIndex and get answers.
It's a console application that you start with a URI to an MLINdex as argument. When you ask a question, it will look up the index to retrieve relevant content and post the question with the relevant content to OpenAI chat model (gpt-3.5-turbo or gpt4) to get an answer.

## How it works?

## Get started
### Create .env file in this folder with below content
```
OPENAI_API_BASE=<AOAI_endpoint>
OPENAI_API_KEY=<AOAI_key>
CHAT_MODEL_DEPLOYMENT_NAME=gpt-35-turbo
PROMPT_TOKEN_LIMIT=3000
MAX_COMPLETION_TOKENS=256
VERBOSE=false
```
Note: CHAT_MODEL_DEPLOYMENT_NAME should point to a chat model like gpt-3.5-turbo or gpt-4

### Run the command line
```shell
python main.py <uri-to-mlindex>
```
