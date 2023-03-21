#!/bin/bash
#SBATCH --job-name=eval-harness-deepspeed
#SBATCH --constraint=v100-16g
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1          # crucial - only 1 task per dist per node!
#SBATCH --cpus-per-task=40           # number of cores per tasks
#SBATCH --hint=nomultithread         # we get physical cores not logical
#SBATCH --gres=gpu:1                 # number of gpus
#SBATCH --time 20:00:00              # maximum execution time (HH:MM:SS)
#SBATCH --output=%x-%j.out           # output file name
#SBATCH --account=six@gpu


set -x -e

source $six_ALL_CCFRWORK/start-prod

echo "START TIME: $(date)"

# a unique identifier for the current eval so that multiple evals could run in parallel and not all log into the same "results.json" file.
VARIANT="tr9c-1B3-swiglu"

CHECKPOINT_PATH=/gpfsdsstore/projects/rech/six/commun/checkpoints/tr3m-1B3-emb-norm-pile/global_step296023
MEGATRON_DEEPSPEED_REPO=/gpfsssd/worksf/projects/rech/six/commun/code/eval/Megatron-DeepSpeed

# you want these 2 on JZ, and pre-download/cache any datasets/tokenizers/models
# but comment these out if you're running on a node with Internet access
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

cd $MEGATRON_DEEPSPEED_REPO

# eval topology
PP_SIZE=1
TP_SIZE=1

VOCAB_FILE=$MEGATRON_DEEPSPEED_REPO/data/gpt2-vocab.json
MERGE_FILE=$MEGATRON_DEEPSPEED_REPO/data/gpt2-merges.txt
SEQ_LEN=2048

# different from the training MICRO_BATCH_SIZE - no optim memory, so can do bigger BS
# make as big as it can fit into gpu w/o OOM, but not too close to 100%

EVAL_MICRO_BATCH_SIZE=6  # 16GB GPU 1.3B model
#EVAL_MICRO_BATCH_SIZE=12 # 32GB GPU 1.3B model


#dummy arguments to make megatron happy.
MEGATRON_REQUIRED_ARGS=" \
    --num-layers -1 \
    --hidden-size -1 \
    --num-attention-heads -1 \
    --seq-length -1  \
    --max-position-embeddings -1
"


ZERO_STAGE=0

config_json="./ds_config.json"
cat <<EOT > $config_json
{
  "train_micro_batch_size_per_gpu": 1,
  "train_batch_size": 1,
  "zero_optimization": { "stage": $ZERO_STAGE },
  "fp16": { "enabled": true },
  "steps_per_print": 2000,
  "wall_clock_breakdown": false
}
EOT

CMD="./tasks/eval_harness/evaluate.py  \
    --load $CHECKPOINT_PATH \
    --results_path $VARIANT-results.json \
    --tensor-model-parallel-size $TP_SIZE  \
    --pipeline-model-parallel-size $PP_SIZE \
    --vocab-file $VOCAB_FILE \
    --merge-file $MERGE_FILE \
    --micro-batch-size $EVAL_MICRO_BATCH_SIZE \
    --no-load-optim \
    --no-load-rng \
    --inference \
    --deepspeed \
    --deepspeed_config ds_config.json \
    --seq-length $SEQ_LEN \
    --adaptive_seq_len \
    --eval_fp32 \
    --task_list arc_challenge,arc_easy,boolq,copa,hellaswag,lambada,logiqa,mathqa,mc_taco,mrpc,multirc,openbookqa,piqa,prost,pubmedqa,qnli,qqp,race,rte,sst,webqs,wic,winogrande,wnli,wsc,triviaqa,sciq \
    $MEGATRON_REQUIRED_ARGS \
    "

N_GPUS=1
LAUNCHER="deepspeed --num_gpus $N_GPUS"
echo $LAUNCHER $CMD

export PYTHONPATH=$MEGATRON_DEEPSPEED_REPO

$LAUNCHER $CMD 2>&1 | tee $VARIANT-eval-harness.log
