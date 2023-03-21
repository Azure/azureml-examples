#!/bin/bash
BASE_DATA_PATH=/tmp
INPUT_PATH=/tmp
OUTPUT_PATH=/tmp

wget https://s3.amazonaws.com/models.huggingface.co/bert/gpt2-vocab.json -P ${BASE_DATA_PATH}
wget https://s3.amazonaws.com/models.huggingface.co/bert/gpt2-merges.txt -P ${BASE_DATA_PATH}

python create_dummy_dataset.py --dir ${INPUT_PATH}

python preprocess_data.py \
    --input ${INPUT_PATH}/dataset_0.json \
    --output-prefix ${OUTPUT_PATH}/dataset-0 \
    --vocab ${BASE_DATA_PATH}/gpt2-vocab.json \
    --dataset-impl mmap \
    --tokenizer-type GPT2BPETokenizer \
    --merge-file ${BASE_DATA_PATH}/gpt2-merges.txt \
    --append-eod

python preprocess_data.py \
    --input ${INPUT_PATH}/dataset_1.json \
    --output-prefix ${OUTPUT_PATH}/dataset-1 \
    --vocab ${BASE_DATA_PATH}/gpt2-vocab.json \
    --dataset-impl mmap \
    --tokenizer-type GPT2BPETokenizer \
    --merge-file ${BASE_DATA_PATH}/gpt2-merges.txt \
    --append-eod

python preprocess_data.py \
    --input ${INPUT_PATH}/dataset_2.json \
    --output-prefix ${OUTPUT_PATH}/dataset-2 \
    --vocab ${BASE_DATA_PATH}/gpt2-vocab.json \
    --dataset-impl mmap \
    --tokenizer-type GPT2BPETokenizer \
    --merge-file ${BASE_DATA_PATH}/gpt2-merges.txt \
    --append-eod

python preprocess_data.py \
    --input ${INPUT_PATH}/dataset_3.json \
    --output-prefix ${OUTPUT_PATH}/dataset-3 \
    --vocab ${BASE_DATA_PATH}/gpt2-vocab.json \
    --dataset-impl mmap \
    --tokenizer-type GPT2BPETokenizer \
    --merge-file ${BASE_DATA_PATH}/gpt2-merges.txt \
    --append-eod

python preprocess_data.py \
    --input ${INPUT_PATH}/dataset_4.json \
    --output-prefix ${OUTPUT_PATH}/dataset-4 \
    --vocab ${BASE_DATA_PATH}/gpt2-vocab.json \
    --dataset-impl mmap \
    --tokenizer-type GPT2BPETokenizer \
    --merge-file ${BASE_DATA_PATH}/gpt2-merges.txt \
    --append-eod

DIR=`pwd`
DATETIME=`date +'date_%y-%m-%d_time_%H-%M-%S'`
mkdir -p ${BASE_DATA_PATH}/logs

DATASET_0="${OUTPUT_PATH}/dataset-0_text_document"
DATASET_1="${OUTPUT_PATH}/dataset-1_text_document"
DATASET_2="${OUTPUT_PATH}/dataset-2_text_document"
DATASET_3="${OUTPUT_PATH}/dataset-3_text_document"
DATASET_4="${OUTPUT_PATH}/dataset-4_text_document"
DATASET="0.1 ${DATASET_0} 0.25 ${DATASET_1} 0.2 ${DATASET_2} 0.15 ${DATASET_3} 0.3 ${DATASET_4}"

VOCAB_PATH=${BASE_DATA_PATH}/gpt2-vocab.json
MERGE_PATH=${BASE_DATA_PATH}/gpt2-merges.txt


#script_path=$(realpath $0)
#script_dir=$(dirname $script_path)
#CONFIG_JSON="$script_dir/ds_config.json"

USE_DEEPSPEED=1
ZERO_STAGE=0


# Debug
#TP=4
#PP=4
#LAYERS=8
#HIDDEN=512
#SEQ=1024
#GLOBAL_BATCH=128
#WORKER_STR="-i worker-0"


# 52B
TP=4
PP=16
HIDDEN=1024
LAYERS=24
SEQ=128
GLOBAL_BATCH=16
WORKER_STR=""

MICRO_BATCH=8

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
	--train-iters 1000 \
        --lr 6.0e-5 \
	--min-lr 6.0e-6 \
        --lr-decay-style cosine \
        --log-interval 1 \
        --eval-iters 40 \
        --eval-interval 1000 \
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
        --fp16 \
	--checkpoint-activations
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

  "gradient_clipping": 1.0,
  "prescale_gradients": true,

  "fp16": {
    "enabled": true,
    "loss_scale": 0,
    "loss_scale_window": 500,
    "hysteresis": 2,
    "min_loss_scale": 1,
    "initial_scale_power": 12
  },

  "wall_clock_breakdown" : true
}
EOT

# run_cmd="deepspeed $WORKER_STR ${DIR}/test_sampling.py $@ ${options}"
run_cmd="deepspeed $WORKER_STR test_sampling.py $@ ${options}"

echo ${run_cmd}
eval ${run_cmd}

set +x
