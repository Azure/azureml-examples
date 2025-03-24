# pylint: disable=line-too-long,useless-suppression
# ------------------------------------
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
# ------------------------------------

"""
DESCRIPTION:
    This sample demonstrates how to use agent operations with code interpreter from
    the Azure Agents service using a synchronous client.

    The sample is for Llama 3.3 70B NIM

USAGE:
    python sample_agents_code_interpreter.py

    Before running the sample:

    pip install azure-ai-projects azure-identity

    Set these environment variables with your own values:
    1) PROJECT_CONNECTION_STRING - The project connection string, as found in the overview page of your
       Azure AI Foundry project.
    2) MODEL_DEPLOYMENT_NAME - The deployment name of the AI model, as found under the "Name" column in 
       the "Models + endpoints" tab in your Azure AI Foundry project.
"""

import os
from azure.ai.projects import AIProjectClient
from azure.ai.projects.models import CodeInterpreterTool, FilePurpose, MessageRole
from azure.identity import DefaultAzureCredential
from pathlib import Path

os.environ["PROJECT_CONNECTION_STRING"] = "<enter-project-connection-string>"
os.environ[
    "MODEL_DEPLOYMENT_NAME"
] = "https://<endpoint name>.<region>.inference.ml.azure.com/v1/@meta/llama-3.3-70b-instruct"


project_client = AIProjectClient.from_connection_string(
    credential=DefaultAzureCredential(),
    conn_str=os.environ["PROJECT_CONNECTION_STRING"],
)

with project_client:
    # Create agent
    agent = project_client.agents.create_agent(
        model=os.environ["MODEL_DEPLOYMENT_NAME"],
        name="my-assistant",
        instructions="You are helpful assistant",
    )
    # [END upload_file_and_create_agent_with_code_interpreter]
    print(f"Created agent, agent ID: {agent.id}")

    thread = project_client.agents.create_thread()
    print(f"Created thread, thread ID: {thread.id}")

    # Create a message
    message = project_client.agents.create_message(
        thread_id=thread.id,
        role="user",
        content="When was the universe founded?",
    )
    print(f"Created message, message ID: {message.id}")

    run = project_client.agents.create_and_process_run(
        thread_id=thread.id, agent_id=agent.id
    )
    print(f"Run finished with status: {run.status}")

    if run.status == "failed":
        # Check if you got "Rate limit is exceeded.", then you want to get more quota
        print(f"Run failed: {run.last_error}")

    # [START get_messages_and_save_files]
    messages = project_client.agents.list_messages(thread_id=thread.id)
    print(f"Messages: {messages}")

    last_msg = messages.get_last_text_message_by_role(MessageRole.AGENT)
    if last_msg:
        print(f"Last Message: {last_msg.text.value}")

    project_client.agents.delete_agent(agent.id)
    print("Deleted agent")
