This code is taken from [llama-recipes](https://github.com/facebookresearch/llama-recipes/tree/main) published by meta and further changed to add the 
following.
- Added support for cpu_offload and meta_tensors(via low_cpu_fsdp flag). This enabled us to finetune 70b on A100 & ND40 machine.
- Support for Image classification task. Tested with [emotion-detection dataset](https://huggingface.co/datasets/dair-ai/emotion)
- Integrated azureml-metrics for evaluation
- Added model.generate for prediction dataset

# Quick Start command job
Follow the FSDP_notebook.ipynb to submit the command job on a compute. 
For Finetuning 70b on ND40, use the following flags.
`low_cpu_fsdp = True
cpu_offload = True
`
# Quick Start Local
1. Create a virtual environment with the requirements mentioned in requirement.txt.
```bash
pip install -r requirements.txt
```

2. Use the following command to run it
```bash
torchrun --nnodes 1 --nproc_per_node <NUM_GPUS>  llama_finetuning.py --enable_fsdp --use_peft --peft_method lora --model_name /patht_of_model_folder/7B --pure_bf16 --output_dir Path/to/save/PEFT/model
```

Here we load the model in bf16(pure_bf16 argument) which only works on high-end skus such as A100. For NC/ND machines, load the model in half precision using `--use_fp16`. 

# Additional Details:
- We are installing the PyTorch Nightlies since FSDP is only support via nightlies.
- Int8 quantization from bit&bytes currently is not supported with FSDP.
- For LLaMA 70b, finetuning with FSDP + PEFT(LoRA) works on A100 machine. On ND40, LLaMA-70b requires atleast 2 ND40 nodes with cpu_offloading. Use the following command for 70b 
```bash
torchrun --nnodes 1 --nproc_per_node 6  llama_finetuning.py --enable_fsdp --use_peft --peft_method lora --model_name /patht_of_model_folder/70B --pure_bf16 --output_dir Path/to/save/PEFT/model
```

# Parameters:
- **use_peft:** Boolean flag to enable Parameter-Efficient Fine-Tuning([PEFT](https://github.com/huggingface/peft)).
- **peft_method:** Method to use for PEFT. Default is LoRA.
- **pure_bf16:** Boolean flag to load the model in bfp16. This only works on A100 machine.
- **use_fp16:** Boolean flag to load the model in fp16 precision.
- **enable_fsdp:** Boolean flag to enable FSDP
- All the setting defined in [config files](./configs/) can be passed as args through CLI when running the script, there is no need to change from config files directly.

### Single GPU:

```bash
#if running on multi-gpu machine
export CUDA_VISIBLE_DEVICES=0

python llama_finetuning.py  --use_peft --peft_method lora --quantization --model_name /patht_of_model_folder/7B --output_dir Path/to/save/PEFT/model

```

### Multiple GPUs One Node:

**NOTE** please make sure to use PyTorch Nightlies for using PEFT+FSDP. Also, note that 

```bash

torchrun --nnodes 1 --nproc_per_node 4  llama_finetuning.py --enable_fsdp --use_peft --peft_method lora --model_name /patht_of_model_folder/7B --pure_bf16 --output_dir Path/to/save/PEFT/model
```

Here we use FSDP for training. To make use of PEFT methods with FSDP make sure to pass `use_peft` and `peft_method` args along with `enable_fsdp`. Here we are using `BF16` for training.

### Fine-tuning using FSDP Only

If you are interested in running full parameter fine-tuning without making use of PEFT methods, please use the following command. Make sure to change the `nproc_per_node` to your available GPUs. This has been tested with `BF16` on 8xA100, 40GB GPUs.

```bash

torchrun --nnodes 1 --nproc_per_node 8  llama_finetuning.py --enable_fsdp --model_name /patht_of_model_folder/7B --dist_checkpoint_root_folder model_checkpoints --dist_checkpoint_folder fine-tuned

```


# Repository Organization
This repository is organized in the following way:

[configs](configs/): Contains the configuration files for PEFT methods, FSDP, Datasets.

[docs](docs/): Example recipes for single and multi-gpu fine-tuning recipes.

[ft_datasets](ft_datasets/): Contains individual scripts for each dataset to download and process. Note: Use of any of the datasets should be in compliance with the dataset's underlying licenses (including but not limited to non-commercial uses)


[inference](inference/): Includes examples for inference for the fine-tuned models and how to use them safely.

[model_checkpointing](model_checkpointing/): Contains FSDP checkpoint handlers.

[policies](policies/): Contains FSDP scripts to provide different policies, such as mixed precision, transformer wrapping policy and activation checkpointing along with any precision optimizer (used for running FSDP with pure bf16 mode).

[utils](utils/): Utility files for:

- `train_utils.py` provides training/eval loop and more train utils.

- `dataset_utils.py` to get preprocessed datasets.

- `config_utils.py` to override the configs received from CLI.

- `fsdp_utils.py` provides FSDP  wrapping policy for PEFT methods.

- `memory_utils.py` context manager to track different memory stats in train loop.
