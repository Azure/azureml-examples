#!/bin/bash


DIR=`pwd`
DATETIME=`date +'date_%y-%m-%d_time_%H-%M-%S'`
#mkdir -p $DIR/logs
#mkdir -p /tmp/logs


#DATASET_1="<PATH TO THE FIRST DATASET>"
#DATASET_2="<PATH TO THE SECOND DATASET>"
#DATASET_3="<PATH TO THE THIRD DATASET>"
#DATASET="0.2 ${DATASET_1} 0.3 ${DATASET_2} 0.5 ${DATASET_3}"

#BASE_DATA_PATH=tests/data/gpt2
#DATASET=${BASE_DATA_PATH}/meg-gpt2-openwebtext_text_document
#VOCAB_PATH=${BASE_DATA_PATH}/gpt2-tiny-vocab.json
#MERGE_PATH=${BASE_DATA_PATH}/gpt2-tiny-merges.txt

BASE_DATA_PATH=/vc_data/Megatron-LM/data
DATASET=${BASE_DATA_PATH}/indexed_datasets/megatron
VOCAB_PATH=${BASE_DATA_PATH}/gpt2-vocab.json
MERGE_PATH=${BASE_DATA_PATH}/gpt2-merges.txt


script_path=$(realpath $0)
script_dir=$(dirname $script_path)
CONFIG_JSON="$script_dir/ds_config.json"
#CONFIG_JSON="/tmp/ds_config.json"

USE_DEEPSPEED=1
ZERO_STAGE=0

#TP=4
#PP=4

# Debug
DEBUG_MODE=0 
if [[ $DEBUG_MODE == 1 ]]; then
        LAYERS=4
        HIDDEN=512
        SEQ=512
        EXIT_INTERVAL=3
else
        HIDDEN=1024
        LAYERS=24
        SEQ=1024
        EXIT_INTERVAL=10
fi  

TP=2
PP=2
DP=4
WORLD_SIZE=$((TP*PP*DP))
GLOBAL_BATCH=4

MICRO_BATCH=1
TRAIN_ITERS=100000
CHECKPOINT_PATH=checkpoints/gpt2/tp${TP}_pp${PP}_dp${DP} 
LOAD_CHECKPOINT_PATH=checkpoints/gpt2/tp${TP}_pp${PP}_dp${DP}

LR=6.0e-4
MIN_LR=6.0e-5
DTYPE="bf16"
EXP_DIR=${HOME}/experiments/results/ckpt_reshape
LOG_DIR="${EXP_DIR}/tensorboard/tp${TP}_pp${PP}_dp${DP}_hd${HIDDEN}_nl${LAYERS}_gbsz${GLOBAL_BATCH}_mbsz${MICRO_BATCH}_z${ZERO_STAGE}_LR_${LR}_${MIN_LR}_${DTYPE}_cont"
mkdir -p $LOG_DIR

while [[ $# -gt 0 ]]
do
key="$1"
case $key in
    --no-deepspeed)
    USE_DEEPSPEED=0;
    shift
    ;;
    -z|--zero-stage)
    ZERO_STAGE=$2;
    shift
    ;;
    *)
    echo "Unknown argument(s)"
    usage
    exit 1
    shift
    ;;
esac
done


options=" \
	--tensor-model-parallel-size $TP \
	--pipeline-model-parallel-size $PP \
        --num-layers $LAYERS \
        --hidden-size $HIDDEN \
        --num-attention-heads 32 \
        --seq-length $SEQ \
        --loss-scale 12 \
        --max-position-embeddings $SEQ \
	--micro-batch-size $MICRO_BATCH \
	--global-batch-size $GLOBAL_BATCH \
	--train-iters $TRAIN_ITERS \
        --lr $LR \
	--min-lr $MIN_LR \
        --lr-decay-style cosine \
        --log-interval 1 \
        --eval-iters 40 \
        --eval-interval 10 \
	--data-path ${DATASET} \
	--vocab-file ${VOCAB_PATH} \
	--merge-file ${MERGE_PATH} \
	--save-interval 1000 \
        --split 98,2,0 \
        --clip-grad 1.0 \
	--weight-decay 0.1 \
	--adam-beta1 0.9 \
	--adam-beta2 0.95 \
	--init-method-std 0.006 \
        --${DTYPE} \
	--checkpoint-activations \
	--exit-interval ${EXIT_INTERVAL} \
        --save ${CHECKPOINT_PATH} \
        --load ${LOAD_CHECKPOINT_PATH} \
        --position-embedding-type alibi \
        --override-lr-scheduler \
        --embed-layernorm \
	--tensorboard-dir $LOG_DIR
        "


if [[ ${USE_DEEPSPEED} -eq 1 ]]; then
	echo "Using DeepSpeed"
	options="${options} \
		--deepspeed \
		--deepspeed_config=${CONFIG_JSON} \
		--zero-stage=${ZERO_STAGE} \
		--deepspeed-activation-checkpointing \
	"
fi


cat <<EOT > $CONFIG_JSON
{
  "train_batch_size" : $GLOBAL_BATCH,
  "train_micro_batch_size_per_gpu": $MICRO_BATCH,
  "steps_per_print": 1,

  "zero_optimization": {
    "stage": $ZERO_STAGE
  },

  "bf16": {
    "enabled": true
  },

  "fp16": {
    "enabled": false,
    "loss_scale": 0,
    "loss_scale_window": 500,
    "hysteresis": 2,
    "min_loss_scale": 1,
    "initial_scale_power": 12
  },

  "wall_clock_breakdown" : true
}
EOT

#WORKER_STR="--num_nodes 1 --num_gpus $WORLD_SIZE"
#WORKER_STR="-i worker-0:0,1,2,3"
#run_cmd="deepspeed -i worker-0:0,1,2,3 ${DIR}/pretrain_gpt.py $@ ${options}"
#run_cmd="deepspeed -i worker-0 ${DIR}/pretrain_gpt.py $@ ${options}"
run_cmd="deepspeed --master_port 29700 $WORKER_STR ${DIR}/pretrain_gpt.py $@ ${options}"


echo ${run_cmd}
eval ${run_cmd}

set +x
