FROM vllm/vllm-openai:latest

ENTRYPOINT [ "python3", "-m", "vllm.entrypoints.openai.api_server", "model", "meta-llama/Meta-Llama-3-8B" ]