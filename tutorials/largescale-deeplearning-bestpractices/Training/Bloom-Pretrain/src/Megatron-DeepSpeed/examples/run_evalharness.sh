CHECKPOINT_PATH=/gpfsscratch/rech/bbv/utw68ny/checkpoints/tr3m-1B3-pile/global_step296023/

PP_SIZE=1
TP_SIZE=1
VOCAB_FILE=gpt2-vocab.json
MERGE_FILE=gpt2-merges.txt

export HF_DATASETS_OFFLINE=1

#dummy arguments to make megatron happy.
MEGATRON_REQUIRED_ARGS="\
    --num-layers -1\
    --hidden-size -1\
    --num-attention-heads -1\
    --seq-length -1 \
    --max-position-embeddings -1
"

CMD="./tasks/eval_harness/evaluate.py \
    --load $CHECKPOINT_PATH\
    --tensor-model-parallel-size $TP_SIZE \
    --pipeline-model-parallel-size $PP_SIZE\
    --vocab-file $VOCAB_FILE\
    --merge-file $MERGE_FILE\
    --micro-batch-size 64\
    --adaptive_seq_len\
    --eval_fp32\
    --task_list hellaswag,mrpc,piqa\
    $MEGATRON_REQUIRED_ARGS\
    "

N_GPUS=1
LAUNCHER="deepspeed --num_gpus $N_GPUS"
$LAUNCHER $CMD