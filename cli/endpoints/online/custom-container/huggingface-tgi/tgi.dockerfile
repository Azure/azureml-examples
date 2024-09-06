FROM ghcr.io/huggingface/text-generation-inference:2.2.0
 
ENV MODEL_ID=meta-llama/Meta-Llama-3.1-8B-Instruct
 
ENTRYPOINT ["tgi-entrypoint.sh", "--model-id", "$MODEL_ID"]
