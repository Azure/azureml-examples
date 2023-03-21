# **Bloom Pretraining on AzureML**

BigScience Large Open-science Open-access Multilingual Language Model (BLOOM), is an autoregressive language model based on the GPT-3 architecture. BLOOM is trained on data from 46 natural languages and 13 programming languages and is the largest publicly available open multilingual model. Training this large model required multiple optimizations to train efficiently. This guide details the process.

## **Setup**
### **Hardware**

NVIDIA A100 80GB GPUs are recommended for this job. This experiment was originally run with 2 Standard_ND96amsr_A100_v4 nodes that have 8 A100 GPUs each. Each node is NVLink enabled.
#### **Linear Scaling with Infiniband Enabled SKUs**
To attain linear scaling for large model, one important step can be to use InfiniBand. InfiniBand enables low-latency, GPU-to-GPU communication across nodes in a cluster. InfiniBand requires specialized hardware to operate. Only some VM SKUs on Azure contain this required hardware. You can view the full list of InfiniBand-enabled machine SKUs [here](https://learn.microsoft.com/en-us/azure/virtual-machines/sizes-hpc#rdma-capable-instances). 

### **Environment**
The environment found [here](./src/environment/context/Dockerfile) is an ACPT environment with multiple accelerators to boost the training job. Also included are HuggingFace packages used for this training. If you would like to add additional packages, edit the appropriate files in that directory with your changes, then create the custom environment using the following command:
```
az ml environment create --file ./src/environment/env.yml
```

## **Code**
The following code can be found under this directory in ``src/deepspeed-BLOOM-AML-SDKv2.yaml`` for the submit file and environment and ``src/Megatron-DeepSpeed/pretrain_gpt.py`` for the training code.

### **Job Configuration**
In the [``deepspeed-BLOOM-AML-SDKv2.yaml``](./src/deepspeed-BLOOM-AML-SDKv2.yaml) file for submitting the job, there are several arguments passed in for the pretraining, with most being settings specific to how the model will be trained. For more information on command line arguments, see [here](https://github.com/bigscience-workshop/Megatron-DeepSpeed/blob/main/megatron/arguments.py). Some arguments relevant to this example are:

- ``--data-path`` - The paths to the data the model is trained on. The format should be the weight of the dataset and the path and name of the file that references .bin and .idx file (without the extension). For example, command below will add weight of 0.033178301 to ar language data, and inside the ar folder should be ar_text_document.bin and ar_text_document.idx.
- ``--deepspeed`` and other deepspeed related arguments. These arguments are specific to DeepSpeed. The ``ds_config.json`` file passed in gives the configuration settings for DeepSpeed. Notice that the argument ``global_batch_size`` matches the ``train_batch_size`` setting in the ds_config. Similarly, the ``--zero_stage`` command line argument matches the ``zero_optimization`` setting in the ``ds_config.json`` file.
- ``--bf16`` - This enables the bf16 optimizer from DeepSpeed. See the optimizations section below for more details.
- ``--micro-batch-size`` This is the batch size for a single step on one GPU. Tweaking this value can help improve performance. See the optimizations section below for more details.

### **Mounted Dataset**
The training data consists of 46 languages and 13 programming languages. For more information on the dataset language proportions and sources, [check this page](https://github.com/bigscience-workshop/bigscience/tree/master/train/tr11-176B-ml#datasets).

#### **Preprocessing**
The dataset is in 1.6TB of de-duplicated pre-processed text, converted into 350B unique tokens. It uses Byte-Pair Encoding (BPE) tokenization, which is also used by other transformer models including GPT, GPT-2, and RoBERTa. Vocabulary size is 250,680 tokens. More info on the BLOOM tokenizer can be found [here](https://huggingface.co/bigscience/bloom#preprocessing).

The Bloom dataset can be configured to be mounted by adding the following in the `outputs` section of the Command Job yaml file. Place them in `outputs` section instead of `inputs` section as Megatron code processes and writes .npy file in the same folder where .bin and .idx files are located.

```
outputs:
  output:
    type: uri_folder
    mode: rw_mount
    path: azureml://datastores/workspaceblobstore/paths/outputs/checkpoint
  blobstore_datadir:
    type: uri_folder
    mode: rw_mount
    path: azureml://datastores/bloomdatastore/paths/bloom-data
```

The path to this mounted folder will then be passed as a command line argument (``--aml-data-download-path``) as described above.

### **Training**
In our training script, ``pretrain_gpt.py``, there is one function called to perform the training:
```
pretrain(train_valid_test_datasets_provider, model_provider, forward_step,
          args_defaults={'tokenizer_type': 'GPT2BPETokenizer'})
```
The arguments of this function each have a purpose:
- ``train_valid_test_datasets_provider`` - This is the function used to prepare the datasets for training. Under the hood, the datasets passed in are combined and readied for training:
```
if args.data_path:
  train_ds, valid_ds, test_ds = build_train_valid_test_datasets(
    data_prefix=args.data_path,
    data_impl=args.data_impl,
    splits_string=args.split,
    train_valid_test_num_samples=train_val_test_num_samples,
    seq_length=args.seq_length,
    seed=args.seed,
    skip_warmup=(not args.mmap_warmup))
```
- ``model_provider`` - This function creates the model to be trained. This is also where DeepSpeed is initialized.

- ``forward_step`` - The forward step function used at each training step.

Inside the actual pretrain function is where the training loop occurs:
```
if args.do_train and args.train_iters > 0:
  iteration = train(forward_step_func,
                    model, optimizer, lr_scheduler,
                    train_data_iterator, valid_data_iterator)
```
## **Training Optimizations**
Training this large model required multiple optimizations to train efficiently. With the architecture, hardware and software in place we were able to start training with 530B data and optimized to higher GPU utillization on a single node and multinode compute with different batch sizes and parallelism. 

### **bf16 optimizer**
For Bloom model training, bf16 optimizer from deepspeed is used for floating point format as an alternative to FP16. BFLOAT16 requires hardware support (e.g., NVIDIA A100) and are about half the size in silicon of a typical FP16 multiplier, and they are eight times smaller than an FP32 multiplier and reduces the memory footprint significantly.

We enable bf16 by passing the `--bf16` flag as a command line argument. We also need to add the setting to our ``ds_config.json`` file:
```
  "bf16": {
    "enabled": true
  }
```

### **Micro Batch Size**
Configure the ``--micro-batch-size`` argument for efficient GPU Utillization to reduce GPU requirement and increase Teraflops and Samples per second.

## **Results**

- TFlops: TFLOPS stands for "trillion floating point calculations per second". Using DeepSpeed can possibly achieve a higher TFLOPS, indicating that compute resources are being utilized more efficiently.

- Training throughput (Samples per second): This metric refers to the number of training examples that can be processed per second during training. With DeepSpeed, you may be able to achieve higher training throughput, which can help speed up the training process.

|Experiment |Model size|GPU Count |	TP|	PP	 | MBS	| TFlops|	Samples per second |	GPU memory Utillized	
|----|----|----|----|----|----|----|----|----|
|1|25B|16|	8|	1|	1|	119.42|	4.173	|69.7%|
|2|20B|8|	8|	1	|1	|117.71	|2.51	|78.5%|
|3|20B|8|	8|	1	|2|	123.52	|2.63	|80.1%|

### **Results Summary**
Based on the above results, to train 200B model we recommend using 80 GPUs with tensor parallel = 8, pipeline parallel=10, GBS=64 and MBS=1. See below for further details on the three experiments.

## **Experiments**
### **Experiment 1: Baseline**
This experiment was run with 25B parameters on 2 nodes to get the baseline metrics for GPU utillization.

bf16, 2 node, 25B model: TP=8, PP=1, NLAYERS=10, NHIDDEN=14400, NHEADS=32, SEQ_LEN=2048, VOCAB_LENGTH=250k, GBS=16
#### **Max TFlops**: 119.42
#### **Samples per second**: 4.173
| iteration | Consumed Samples | Consumed Tokens | Elapsed time/iteration | Learning rate | global batch size | lm loss | grad norm | num zeros | number of skipped iterations | number of nan iterations | Samples/second |
|---|---|---|---|---|---|---|---|---|---|---|---|
9/100000 | 144 | 294912 | 3.83 | 6.000E-05 | 16 | 1.307423E+01 | 110.837 | 0.0 | 0 | 0 | 4.173 |

### **Monitoring**
#### **Memory Utillization**

<img src="https://user-images.githubusercontent.com/39776179/220196056-b7ee55df-9f53-467b-b145-4e0b6a21d385.png" alt="resource utilization" width="600"/>

#### **GPU Utillization**

<img src="https://user-images.githubusercontent.com/39776179/220196415-a955f68f-8424-4d77-b9a7-26a94d5fb924.png" alt="resource utilization" width="600"/>

#### **GPU Memory Usage**

<img src="https://user-images.githubusercontent.com/39776179/220196510-9d2a01e6-9cd4-4db7-a602-000b578e1f9e.png" alt="resource utilization" width="600"/>

#### **Disk Usage**

<img src="https://user-images.githubusercontent.com/39776179/220196244-73497d35-1a24-4b8f-8ca7-9a28f9772777.png" alt="resource utilization" width="600"/>

#### **IB**

<img src="https://user-images.githubusercontent.com/39776179/220196192-fede4607-8f5b-4dac-8a7c-d77812aaedf2.png" alt="resource utilization" width="600"/>

### **Experiment 2**
This experiment was run using only a single node as the previous experiment had only used 69% of GPU. After testing with a single node the model size was also reduced to 20B parameters due to an out of memory error. Below is the experiment configuration.

bf16, 1 node, 20B model: TP=8, PP=1, NLAYERS=8, NHIDDEN=14400, NHEADS=32, SEQ_LEN=2048, VOCAB_LENGTH=250k, MBS=1, GBS=16
#### **Max TFlops**: 117.71
#### **Samples per second**: 2.510
| iteration | Consumed Samples | Consumed Tokens | Elapsed time/iteration | Learning rate | global batch size | lm loss | grad norm | num zeros | number of skipped iterations | number of nan iterations | Samples/second |
|---|---|---|---|---|---|---|---|---|---|---|---|
352/100000 | 5632 | 11534336 | 6.37 | 6.000E-05 | 16 | 7.416743E+00 | 2.579 | 0.0 | 0 | 0 | 2.510 |

### **Monitoring**
#### **Memory Utillization**

<img src="https://user-images.githubusercontent.com/39776179/220197046-110ca95e-7be1-420b-8496-044d55ee352c.png" alt="resource utilization" width="600"/>

#### **GPU Utillization**

<img src="https://user-images.githubusercontent.com/39776179/220197157-85d04f13-94f8-4335-a9d9-ee3175c1babf.png" alt="resource utilization" width="600"/>

#### **GPU Memory Usage**

<img src="https://user-images.githubusercontent.com/39776179/220197230-681f0e6c-1cb6-436b-99d8-2e1cf3ce63ee.png" alt="resource utilization" width="600"/>

#### **Disk Usage**

<img src="https://user-images.githubusercontent.com/39776179/220197310-b266d096-c5f4-4442-8aaa-383eef2887a5.png" alt="resource utilization" width="600"/>

#### **IB**

<img src="https://user-images.githubusercontent.com/39776179/220197509-82b489e1-40ed-4268-8f0d-4dadaee12092.png" alt="resource utilization" width="600"/>

### **Experiment 3**
This experiment was run on a single node with an increased micro batch size of 2 for higher GPU utillization.

bf16, 1 node, 20B model: TP=8, PP=1, NLAYERS=8, NHIDDEN=14400, NHEADS=32, SEQ_LEN=2048, VOCAB_LENGTH=250k, MBS=2, GBS=2048
#### **Max TFlops**: 123.52
#### **Samples per second**: 2.634
| iteration | Consumed Samples | Consumed Tokens | Elapsed time/iteration | Learning rate | global batch size | lm loss | grad norm | num zeros | number of skipped iterations | number of nan iterations | Samples/second |
|---|---|---|---|---|---|---|---|---|---|---|---|
352/  100000 | 5632 | 11534336 | 6.07 | 6.000E-05 | 16 | 7.515875E+00 | 3.246 | 0.0 | 0 | 0 | 2.634 |
### **Monitoring**
#### **Memory Utillization**

<img src="https://user-images.githubusercontent.com/39776179/220198411-48c626c9-30de-4e97-a220-03bcf1cf5ad2.png" alt="resource utilization" width="600"/>

#### **GPU Utillization**

<img src="https://user-images.githubusercontent.com/39776179/220198555-f28ea3b5-f949-4d9f-8314-823204e4ea95.png" alt="resource utilization" width="600"/>

#### **GPU Memory Usage**

<img src="https://user-images.githubusercontent.com/39776179/220198638-29969e26-b5f0-4894-94de-3a62cd3e5300.png" alt="resource utilization" width="600"/>

#### **Disk Usage**

<img src="https://user-images.githubusercontent.com/39776179/220198733-b27e81ea-a3ec-44bf-9eba-c659785f8896.png" alt="resource utilization" width="600"/>

#### **IB**

<img src="https://user-images.githubusercontent.com/39776179/220198812-07928712-c114-4f4c-b46a-d63cf390fe3e.png" alt="resource utilization" width="600"/>

