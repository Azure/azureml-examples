# Fast Setup instructions

This quick instructions document contains 3 steps:

1. installing software
2. preparing data
3. running the script

This is useful if you need to ask someone to reproduce problems with `Megatron-Deepspeed`

## 1. Software

Please follow this exact order.


0. Create a new conda env if need be or activate an existing environment.

1. Install `pytorch`. Choose the desired version install instructions [here](https://pytorch.org/get-started/locally/), but for conda it'd be:

```
conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch
```

2. Install system-wide `cuda` if you don't have it already. [NVIDIA instructions](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html). Of course ideally use [the premade packages for your distro](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html#package-manager-installation).
Use the same major version as pytorch's cuda build. To check use:

```
python -c 'import torch; print(f"pt={torch.__version__}, cuda={torch.version.cuda}")'
```

The minor versions don't actually have to match, but then you will need to hack `apex` installer to ignore minor version changes, see below.

3. Install `apex`

```
git clone https://github.com/NVIDIA/apex
cd apex
pip install --global-option="--cpp_ext" --global-option="--cuda_ext" --no-cache -v --disable-pip-version-check .  2>&1 | tee build.log
cd -
```

If the pytorch and system-wide cuda minor versions mismatch, it's not a problem, you just need to hack `apex`'s build to bypass the check by applying this patch first and then build it.
```
diff --git a/setup.py b/setup.py
index d76e998..f224dae 100644
--- a/setup.py
+++ b/setup.py
@@ -31,6 +31,8 @@ def check_cuda_torch_binary_vs_bare_metal(cuda_dir):
     print(raw_output + "from " + cuda_dir + "/bin\n")

     if (bare_metal_major != torch_binary_major) or (bare_metal_minor != torch_binary_minor):
+        # allow minor diffs
+        if bare_metal_minor != torch_binary_minor: return
         raise RuntimeError(
             "Cuda extensions are being compiled with a version of Cuda that does "
             "not match the version used to compile Pytorch binaries.  "
```


4. Checkout and prepare `Megatron-DeepSpeed` and install its requirements

```
git clone https://github.com/bigscience-workshop/Megatron-DeepSpeed
cd Megatron-DeepSpeed
pip install -r requirements.txt
```




## 2. Data

Will work under the `Megatron-DeepSpeed` clone

```
cd Megatron-DeepSpeed
```



Prepare data for preprocessing
```
mkdir -p data
wget https://s3.amazonaws.com/models.huggingface.co/bert/gpt2-vocab.json -O data/gpt2-vocab.json
wget https://s3.amazonaws.com/models.huggingface.co/bert/gpt2-merges.txt -O data/gpt2-merges.txt
python -c 'from datasets import load_dataset; ds = load_dataset("stas/oscar-en-10k", split="train", keep_in_memory=False); ds.to_json(f"data/oscar-en-10k.jsonl", orient="records", lines=True, force_ascii=False)'
```

Pre-process a small dataset to be used for training

```
python tools/preprocess_data.py \
    --input data/oscar-en-10k.jsonl \
    --output-prefix data/meg-gpt2-oscar-en-10k \
    --dataset-impl mmap \
    --tokenizer-type GPT2BPETokenizer \
    --merge-file data/gpt2-merges.txt \
    --vocab data/gpt2-vocab.json \
    --append-eod \
    --workers 4
```

now you have data/meg-gpt2-oscar-en-10k, vocab and merges files to pass as arguments to training, the next section shows how to use them.

Note that Megatron wants `data/meg-gpt2-oscar-en-10k_text_document` prefix later in `--data-path`

## 3. Train

Here is a tiny model training setup configured over 2 gpus to train on the data we prepared in step 2.

Put it in a script or run it directly.

If you have only 1 gpu, change these 2 lines below to:

```
N_GPUS=1
TP_SIZE=1
```

The script:

```
CHECKPOINT_PATH=checkpoints/gpt2

VOCAB_FILE=data/gpt2-vocab.json
MERGE_FILE=data/gpt2-merges.txt
DATA_PATH=data/meg-gpt2-oscar-en-10k_text_document
TENSORBOARD_PATH=output_dir/tensorboard

N_GPUS=2
MICRO_BATCH_SIZE=1
GLOBAL_BATCH_SIZE=16
TP_SIZE=2
PP_SIZE=1

NLAYERS=2
NHIDDEN=8
NHEADS=2
SEQ_LEN=512
VOCAB_SIZE=50257

SAVE_INTERVAL=50

TRAIN_SAMPLES=10_000

GPT_ARGS=" \
    --num-layers $NLAYERS \
    --hidden-size $NHIDDEN \
    --num-attention-heads $NHEADS \
    --seq-length $SEQ_LEN \
    --max-position-embeddings $SEQ_LEN \
    --micro-batch-size $MICRO_BATCH_SIZE \
    --rampup-batch-size 2 2 1_000 \
    --global-batch-size $GLOBAL_BATCH_SIZE \
    --train-samples $TRAIN_SAMPLES \
    --optimizer adam \
    --adam-beta1 0.9 \
    --adam-beta2 0.95 \
    --adam-eps 1e-8 \
    --lr 1e-4 \
    --lr-warmup-samples 5 \
    --min-lr 1e-6 \
    --lr-decay-style cosine \
    --lr-decay-samples 12 \
    --clip-grad 1.0 \
    --weight-decay 1e-1 \
    --embed-layernorm \
    --fp16 \
    --partition-activations \
    --seed 42 \
    --vocab-file $VOCAB_FILE \
    --merge-file $MERGE_FILE \
    "

OUTPUT_ARGS=" \
    --exit-interval 100 \
    --log-interval 10 \
    --save-interval $SAVE_INTERVAL \
    --eval-interval 100 \
    --eval-iters 10 \
    --checkpoint-activations \
    "

DATA_ARGS=" \
    --save $CHECKPOINT_PATH \
    --load $CHECKPOINT_PATH \
    --data-path $DATA_PATH \
    --tensorboard-dir $TENSORBOARD_PATH \
    --tensorboard-queue-size 5 \
    --log-timers-to-tensorboard \
    --log-batch-size-to-tensorboard \
    --log-validation-ppl-to-tensorboard \
    --kill-switch-path /tmp/kill-switch \
    "

ZERO_STAGE=1

config_json="./ds_config.json"

# Deepspeed figures out GAS dynamically from dynamic GBS via set_train_batch_size()
cat <<EOT > $config_json
{
  "train_micro_batch_size_per_gpu": $MICRO_BATCH_SIZE,
  "train_batch_size": $GLOBAL_BATCH_SIZE,
  "gradient_clipping": 1.0,
  "zero_optimization": {
    "stage": $ZERO_STAGE
  },
  "fp16": {
    "enabled": true,
    "loss_scale": 0,
    "loss_scale_window": 500,
    "hysteresis": 2,
    "min_loss_scale": 1,
    "initial_scale_power": 12
  },
  "steps_per_print": 2000,
  "wall_clock_breakdown": false
}
EOT

DEEPSPEED_ARGS=" \
    --deepspeed \
    --deepspeed_config ${config_json} \
    --zero-stage ${ZERO_STAGE} \
    --deepspeed-activation-checkpointing \
    "

ALL_ARGS="$GPT_ARGS $OUTPUT_ARGS $DATA_ARGS $DEEPSPEED_ARGS"

MASTER_ADDR=localhost
MASTER_PORT=6777

export LAUNCHER="python -u -m torch.distributed.run \
    --nproc_per_node $N_GPUS \
    --nnodes 1 \
    --rdzv_endpoint $MASTER_ADDR:$MASTER_PORT \
    --rdzv_backend c10d \
    --max_restarts 0 \
    --tee 3 \
    "
export CMD=" \
    $LAUNCHER pretrain_gpt.py \
    --tensor-model-parallel-size $TP_SIZE \
    --pipeline-model-parallel-size $PP_SIZE \
    --distributed-backend nccl \
    $ALL_ARGS \
    "

echo $CMD

$CMD

```

You can, of course, run this as a slurm script, but here is [a full slurm script example](https://github.com/bigscience-workshop/bigscience/blob/d57b76bb592832bb4d2054cd5cbf132796be2d83/train/tr11-176B-ml/setup-test-n2.slurm), which has some tweaks to get `MASTER_ADDR` and a few other bits right under the SLURM environment on JeanZay, which may or may not be needed if you run it elsewhere.

Remember to wipe out `$CHECKPOINT_PATH`, if you change the model shape and there is a checkpoint with the old shapes saved already.
