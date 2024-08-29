FROM vllm/vllm-openai:latest
 
ENV MODEL_NAME="microsoft/Phi-3-mini-4k-instruct"
 
ENTRYPOINT [ "python3", "-m", "vllm.entrypoints.openai.api_server", "model", "$MODEL_NAME" ]