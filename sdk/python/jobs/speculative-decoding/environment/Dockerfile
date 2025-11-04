FROM lmsysorg/sglang:v0.5.2rc2-cu126
RUN apt-get update && apt-get install -y runit
ENV BASE_MODEL nvidia/Llama-3.1-8B-Instruct-FP8
ENV DRAFT_MODEL lmsys/sglang-EAGLE3-LLaMA3.1-Instruct-8B
ENV SGLANG_ARGS "--tp-size 1 --max-running-requests 32 --mem-fraction-static 0.8 --enable-torch-compile --speculative-algorithm EAGLE3 --speculative-num-steps 3 --speculative-eagle-topk 2 --speculative-num-draft-tokens 4 --dtype float16 --attention-backend fa3 --host 0.0.0.0 --port 30000"
ENV SGL_HOST 0.0.0.0 
ENV SGL_PORT 30000
ENV SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN 1

EXPOSE 30000
ENTRYPOINT python3 -m sglang.launch_server --model-path $BASE_MODEL --speculative-draft-model-path $DRAFT_MODEL $SGLANG_ARGS
