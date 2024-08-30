FROM ghcr.io/huggingface/text-generation-inference:2.2.0

ENV MODEL_ID=microsoft/Phi-3-mini-4k-instruct

CMD ["--model-id", "$MODEL_ID"]
ENTRYPOINT ["tgi-entrypoint.sh"]
