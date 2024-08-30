FROM vllm/vllm-openai:latest
 
ENTRYPOINT [ "python3", "-m", "vllm.entrypoints.openai.api_server", "model", "microsoft/Phi-3-mini-4k-instruct" ]