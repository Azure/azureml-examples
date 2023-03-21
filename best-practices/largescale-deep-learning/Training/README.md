# Large Scale Distributed Training

Large scale training has led to state-of-the-art accuracies across a range of tasks and numerous customers have been using Azure Machine Learning for training models with millions/billions of parameters. While large scale training has led to high accuracies, it also comes with challenges. 

  - GPU memory capacity is limited, making it impossible to fit large models on a single GPU or even on a multi-GPU server.

  - Number of compute operations required to train large models can result in long training times.
  
This guide will show best practices to allow you to train large models very efficiently with high throughput in AzureML, leveraging full utilization of GPU to keep the cost low.

- [Large Scale Distributed Training](#large-scale-distributed-training)
  - [Setup](#setup)
      - [**Linear Scaling with Infiniband Enabled SKUs**](#linear-scaling-with-infiniband-enabled-skus)
  - [Optimizations](#optimizations)
      - [**DeepSpeed Autotuning**](#deepspeed-autotuning)
  - [Monitoring](#monitoring)
- [Create path for logging to tensorboard](#create-path-for-logging-to-tensorboard)
  - [**Examples**](#examples)
      
<!-- /TOC -->

## Setup
- ### **Estimate Memory Requirements**
  For a large training job, its improtant to know how much memory is required by model params, gradients and optimizer states. In addition, you will also need enough memory to fit activation calculations and any temporary memory for intermediate calculations, which for long sequences could be significant. Here is estimated calculation for Model using FP16 and Adam optimizers
  ```
    FP16 parameter: 2 bytes

    FP16 Gradient: 2 bytes

    Optimizer state comprise of FP32 variance and FP32 momentum : 8 bytes

    FP32 Parameter for Optimizer Apply: 4 bytes

    FP32 Gradient for Optimizer Apply: 4 bytes
    ------------------------------------------------------------------------------------------------------------------------------------------------------------------

    Estimated Memory for single Model replica: 20Bytes x number of parameters + memory for activations generated
  ```
  ([API to estimate memory usage for model state consumption, but not activations](https://deepspeed.readthedocs.io/en/latest/memory.html)) from DeepSpeed with several accelarations/optimizations to reduce GPU memory. 
  

- ### **Compute Cluster**

  Ideally, as the number of VMs training a given model increases, the time to train that model should decrease linearly. For instance, if training a model using one VM takes 100 seconds, then training that same model using two VMs should ideally take 50 seconds. Also ideally, model quality / accuracy should not be affected by the number of VMs used. To attain linear scaling, one important step is to use InfiniBand. Linear scaling is ideal, but unfortunately as the number of machines increases, communication cost among the nodes also increases. Infiniband can help offset this cost and increase throughput.

  #### **Linear Scaling with Infiniband Enabled SKUs**

  AzureML offers optimized supercomputer hardware with high bandwidth interconnects to enable low latency, GPU-to-GPU communication across nodes in a cluster.
	These GPUs within a node are connected by NVLink and NVSwitch, GPUs across nodes connected by NVIDIA Mellanox 200Gbps Infiniband cards providing 2.8 exaflop/s of peak AI performance in aggregate.

  You can view the full list of InfiniBand-enabled machine SKUs [here](https://learn.microsoft.com/en-us/azure/virtual-machines/sizes-hpc#rdma-capable-instances).

  Let's do an experiment to see how InfiniBand affects the training time as we scale up the number of nodes. First we create two compute clusters. 
  ```
    infiniband_cluster = create_compute_cluster(
        name='nc24rsv3',
        sku='STANDARD_NC24RS_V3',
        workspace=workspace)

    standard_cluster = create_compute_cluster(
        name='nc24sv3',
        sku='STANDARD_NC24S_V3',
        workspace=workspace)
  ```
  Next we kick off two training runs, each training the model on the same dataset for one epoch. Below is the result where Infiniband helps attain linear scaling. 

  |![image](https://user-images.githubusercontent.com/39776179/224841168-88598d2b-85d7-43d6-b7d2-1de50f26ccde.png "Training with Infiniband hardware")| ![image](https://user-images.githubusercontent.com/39776179/224841405-07dfc7f8-ef2c-43d2-b446-42d591635372.png "Training with Standard hardware")|
  |-----------------------------------|---------------------------------|
  | Training with Infiniband hardware | Training with Standard hardware |
  
  > [!NOTE]
  > While InifiBand helps attain linear scaling, there are other reasons/factors that can impact linear scaling and you will see in the document below and solution.. 
  
  
- ### **Environment**
  The recommended environment for a large scale distributed training job is an Azure Container for PyTorch (ACPT) environment with several built in optimizers and is 	described in more detail [here](../Environment/ACPT.md). This environment is built and ready to use under the 	'Environments' tab in AzureML studio. Some optimizers included in the environment are: 
	- Onnx Runtime, Built-in optimizations that deliver up to 1.4X faster training
	- Deepspeed allows to train trillion model parameter at low cost by achieving excellent system throughput and efficiently scale to thousands of GPUs
	- MSCCL, an inter-accelerator communication framework that is built on top of NCCL
	- Nebula, a new fast checkpointing feature to save your checkpoint 1000 times faster


- ### **Data Loading**
  To load data in the most efficient way with large scale distributed training jobs, follow [this guide](../Data-loading/data-loading.md).
## Optimizations
To achive the best possible performance and resource utilization of jobs on AzureML, we employ several different optimization tools showcased below.
- ### **DeepSpeed**

  [DeepSpeed](https://github.com/microsoft/DeepSpeed) is an open-source library developed by Microsoft that optimizes the training of large deep learning models. It aims to reduce the time and memory requirements needed for training large models with trillions of parameters on distributed GPU clusters.

  Deepspeed is based on architecture of zero redundancy optimizer and leverages the aggregate computation and memory resources of data parallelism to reduce the memory and compute requirements of each device (GPU) used for model training. It also reduces the memory consumption of each GPU by partitioning the various model training states (weights, gradients, and optimizer states) across the available devices (GPUs and CPUs) in the distributed training hardware.

  Large Models need Parallelism

  ![image](https://user-images.githubusercontent.com/39776179/225104309-ffa21fab-4a55-4b5a-bc46-07d6d7820baa.png)

  **ZeRO Stage 1**: The optimizer states (e.g., for Adam optimizer, 32-bit weights, and the first, and second moment estimates) are partitioned across the processes, so that each process updates only its partition and can optimize the memory usage.

  **ZeRO Stage 2**: The reduced 32-bit gradients for updating the model weights are also partitioned such that each process retains only the gradients corresponding to its portion of the optimizer states and can speed up the compute efficiency in addition to memory optimization.

  **ZeRO Stage 3**: The 16-bit model parameters are partitioned across the processes. ZeRO-3 will automatically collect and partition them during the forward and backward passes.

  In addition, ZeRO-3 includes the infinity offload engine to form ZeRO-Infinity ([paper](https://arxiv.org/abs/2104.07857)), which can offload all model states to both CPU and NVMe memory for huge memory savings and can train a trillion parameter model on single GPU without running out of memory.

  ![image](https://user-images.githubusercontent.com/39776179/224893244-98d6c487-b3d9-4970-9738-e12c2732611e.png)

  Suggestion on stage to use based on model parameter size to get best optimization for speed and memory
  | Model Size | Optimal DeepSpeed Stage |
  | :----------: | :--------: | 
  |  <300M  |   Stage 0    |
  |     <1B     |   Stage 1    |
  | <10B  |   Stage 2    |
  |  Others  |   Stage 3 (only use cpu_offload if not enough GPUs to fit)    |

  DeepSpeed features can be enabled, disabled, or configured using a config JSON file that should be specified as args.deepspeed_config. 

  To include DeepSpeed in a job using the HuggingFace ``Trainer`` class, simply include the argument ``--deepspeed ds_config.json`` as part of the ``TrainerArguments`` class passed into the Trainer. Example code for Bert Pretraining with Deepspeed and the HuggingFace Trainer class is shown at [BERT pretraining guide](./Bert-Pretrain).
  
  To include DeepSpeed in a job using a custom training loop, DeepSpeed will have to be initialized before the training loop as shown here:

  <img src="https://user-images.githubusercontent.com/73311224/225169813-73766942-cd8b-4c57-8993-eee24c762f0b.png" alt="Enable the ssh on compute." width="500"/>

  An example showing this implementation can be found [here](https://github.com/Azure/azureml-examples/tree/main/cli/jobs/deepspeed/deepspeed-training).
  For a full set of DeepSpeed features see this [API doc](https://www.deepspeed.ai/docs/config-json/).

  #### **DeepSpeed Autotuning**
  When running a job with DeepSpeed, it is always necessary to include a ``ds_config.json`` file that has the configurations that DeepSpeed will use for training. However, it is hard to know what settings are best in your scenario. This is where Autotuning comes in. [DeepSpeed Autotuning](https://www.deepspeed.ai/tutorials/autotuning/) will find the most optimal configuration file that will maximize the training speed and memory efficiency of a model for a given hardware configuration. This can give users the best possible performance, without having to spend time manually tweaking hyperparameters. There are three configurations in particular that Autotuning will help find the best settings for:
  - ``train_micro_batch_size_per_gpu`` - The batch size for a single step on a GPU.
  - ``gradient_accumulation_steps``- Number of training steps to accumulate gradients before using them to compute variables. Increasing this allows for training on bigger batch sizes.
  - ``zero_optimization`` - The DeepSpeed ZeRO stage setting.

  The table below shows the throughput (samples per second) comparison when run on 16 Nvidia V100 GPUs. The corresponding train micro-batch size per GPU (mbs or tmbspg) and ZeRO stage used to achieve the throughput value is also shown in the parentheses. Assume the strategy users would use in the handtuning process is to start from `mbs = 1` and increase mbs by 2 each time until running out of GPU memory.
  - `baseline` is the vanila HF without DeepSpeed (DS) and mbs is hand-tuned.
  - `HF + DS hand-tuned` is HF with DS, and mbs is hand-tuned while other DS configuration uses default values.
  - `HF + DS autotuning` is HF with DS, and the DS configuration is selected from autotuning.

  Notation: Hugging Face (HF), DeepSpeed (DS), ZeRO stage (z), gradient accumulation steps (gas), train micro-batch size per GPU (mbs or tmbspg).

  | Model   name | num_params |     baseline (vanila HF)      |          HF + DS hand-tuned          | HF + DS autotuning (fast-mode) | throughput improvement over baseline | autotuning time (mins) | number of experiments |
  | :----------: | :--------: | :---------------------------: | :----------------------------------: | :----------------------------: | :----------------------------------: | :--------------------: | :-------------------: |
  |  BERT-large  |   0.34B    |  742.692 (gas = 1,mbs = 64)   |  766.929 (z = 1, gas = 1, mbs = 64)  |   808.168 (z1_gas1_tmbspg93)   |                1.09x                 |           36           |          22           |
  |     GPT2     |   0.12B    |   284.142 (gas = 1,mbs = 8)   |  397.827 (z = 1, gas = 1, mbs = 8)   |   431.586 (z1_gas1_tmbspg14)   |                1.52x                 |           25           |          17           |
  | GPT2-medium  |   0.35B    |   71.61 (gas = 1, mbs = 2)    |  142.211 (z = 1, gas = 1, mbs = 4)   |    163.3 (z1_gas1_tmbspg6)     |                 2.28                 |           15           |          25           |
  |  GPT2-large  |   0.77B    |   27.874 (gas = 1, mbs = 1)   |   56.797 (z = 1, gas = 1, mbs = 2)   |    69.061 (z = 1, mbs = 3)     |                2.48x                 |           27           |          13           |
  |   GPT2-xl    |    1.5B    |         Not runnable          |      27.462 (gas = 1, mbs = 1)       |    27.497 (z1_gas1_tmbspg1)    |                 inf                  |           21           |           9           |
  |   DeBERTa    |    1.5B    |         Not runnable          |   140.587 (z = 1, gas = 1 mbs = 8)   |  162.395  (z1_gas1_tmbspg11)   |                 inf                  |           40           |          12           |


  To learn how to use DeepSpeed Autotuning with AzureML, see [this tutorial](./DeepSpeed-Autotuning/README.md).

  When running the Bloom and BERT examples in this repo, the following results were found:
  |      Metrics      |   Vanilla Pytorch     | DeepSpeed + Autotuning|
  | ----------------- | --------------- | ------------------ |
  | Training Time     |      351.75 s   |   253.79 s     |
  | samples/second    |      2431.02    |   3369.37      |
- ### **Onnx Runtime (ORT)**

  In addition to DeepSpeed, we can also use the HuggingFace [Optimum](https://huggingface.co/docs/optimum/index) library and [Onnx Runtime](https://onnxruntime.ai/docs/) to optimize our training. ORT can provide several benefits to a training job, including flexibility with different hardware configurations, memory optimizations that allow fitting of larger models compared to base Pytorch. More details on how exactly Onnx Runtime improves training time and throughput can be found [here](https://huggingface.co/blog/optimum-onnxruntime-training).

  The chart below shows the training acceleration achieved for different models by using ORT in combination with DeepSpeed.

  <img src="https://user-images.githubusercontent.com/73311224/225122922-a02efc89-91dc-4b15-bcbd-e449432fa573.png" alt="Enable the ssh on compute." width="600"/>
  
  Adding Onnx Runtime acceleration to our job can be done with just a few small code changes:
  ```
  from optimum.onnxruntime import ORTTrainer, ORTTrainingArguments

  parser = HfArgumentParser(ORTTrainingArguments)
  ...
  trainer = ORTTrainer(
      model=model,
      args=training_args,
  ...
  ```
  When using the Huggingface Transformers library, the ``Trainer`` class will train the model with the ``train()`` method. This method loops through a typical training loop and can be customized via arguments from the ``TrainingArguments`` class. To use ORT and the accelerations provided by it for our job, we replace these Transformers classes with the ORT equivalents: ``ORTTrainer`` and ``ORTTrainingArguments`` from HuggingFace Optimum.

  In additon to these code changes, we also add an argument to be passed into ``ORTTrainingArguments``:
  ```
  --optim adamw_ort_fused
  ```
  This is an extra argument added with ORTTrainingArguments that applies the Fused Adam Optimizer to give a little extra performance gain. For a training example that uses ORT, See the [BERT Pretrain example](./Bert-Pretrain/README.md).
## Monitoring
- ### **Interactive Debugging**
  Machine learning model training is usually an iterative process and requires significant experimentation. With the Azure Machine Learning interactive job experience, we can access the container where the job is running and iterate on training scripts, monitor progress and even debug the job remotely on local machines.  
  
  Depending on the tool you want to use, add the corresponding service to your Azure cli v2 command job yaml file:
    ```
  services:
    my_jupyterlab:
      job_service_type: jupyter_lab
      nodes: all
    my_tensorboard:
      job_service_type: tensor_board
      log_dir: "outputs/runs/" #default is working directory on job container
      nodes: all
    my_vscode:
      job_service_type: vs_code
      nodes: all
  ```
  
  To access these services once the job starts, go to the job overview page and click on ``Monitor and Debug``. This will open a sidebar page like the one in the image below, showing links to JupyterLab, TensorBoard and VSCode.

  <img src="https://user-images.githubusercontent.com/73311224/225147928-865bb51f-12ba-44c0-80e1-0d26d067f2cf.png" alt="SSH Connections" width="450"/>

  For an example that enables these tools, see [here](./Bert-Pretrain/README.md).

  #### **JupyterLab**
  With JupyterLab, you can open a terminal and interact with the job container as well as iterate on your training script.

  <img src="https://user-images.githubusercontent.com/73311224/225483980-b53800af-0a27-49f4-a8f5-e2328ed94a5a.png" alt="JupyterLab" width="650"/>

  #### **VSCode**
  VSCode can also interact with the job container, but in addition has the added benefit of interactive debugging.

  <img src="https://user-images.githubusercontent.com/73311224/225445237-7c9ba264-abda-47c3-a662-29ae37dfa0d9.png" alt="Tensorboard" width="650"/>

  #### **Tensorboard**

  With TensorBoard we can monitor metrics while the job is running. It also can show resource utilization via Pytorch Profiler (more on this later).

  <img src="https://user-images.githubusercontent.com/73311224/225138261-21a4b2fc-6005-41be-a5fe-d543058c4365.png" alt="Tensorboard" width="650"/>

  > If you are running with DeepSpeed and not just vanilla Pytorch, make sure to also include the following configuration in you ``ds_config.json`` file so that TensorBoard logs DeepSpeed metrics as well.
  >```  
  >"tensorboard": {
  >  "enabled": true,
  >  "output_path": "/output/runs/",
  >  "job_name": "train_bert"
  >},
  >```
  For more information on interacting with jobs, see [this page](https://learn.microsoft.com/en-us/azure/machine-learning/how-to-interactive-jobs?tabs=ui).
- ### **Pytorch Profiler**
  With how long training times can be and how little resources may be available for a large scale training job, it is important to monitor resource utilization. For a clear and consise way to do this while a job is running, we can use the Pytorch Profiler.

  If you are using the HuggingFace Transformers library in your training script, one way to start using the profiler is to use a custom HuggingFace trainer callback.
  ```
  # Create path for logging to tensorboard
  my_logs=os.environ['PWD']+args.tensorboard_log_dir

  class ProfilerCallback(TrainerCallback):
      def on_train_begin(self, args, state, control, model=None, **kwargs):
          self.prof = profiler.profile(
              schedule=profiler.schedule(wait=2, warmup=1, active=3, repeat=2),
              activities=[profiler.ProfilerActivity.CPU, profiler.ProfilerActivity.CUDA],
              on_trace_ready=profiler.tensorboard_trace_handler(my_logs),
              record_shapes=True,
              with_stack=True,
              profile_memory=True
          )
          self.prof.start()

      def on_train_end(self, args, state, control, model=None, **kwargs):
          self.prof.stop()
      
      def on_step_end(self, args, state, control, model=None, **kwargs):
          self.prof.step()
  ```
  > NOTE: To make sure the Pytorch Profiler is visible with Tensorboard, we create a variable called `my_logs` (as shown in the above code) from passing an additional argument ``--tensorboard_log_dir "/outputs/runs/"`` to our training script. This path matches the ``logDir`` property under ``my_tensorboard`` in our yaml file for submitting the job.
  See the [BERT Pretrain example](./Bert-Pretrain/README.md) for the full implementation of this code.

  After the job starts running, go to the TensorBoard as described above and click on 'Pytorch Profiler'. This page will show the relevant resource utilization information.
  
  <img src="https://user-images.githubusercontent.com/73311224/225250070-ee2403f5-0ac4-4543-aa64-faa8c0a5d397.png" alt="Tensorboard" width="650"/>

  If you are not using the HuggingFace Transformers ``Trainer`` class in your training script and instead using your own training loop, try [this tutorial](https://pytorch.org/tutorials/intermediate/tensorboard_profiler_tutorial.html).


- ### **Flops Profiler**

  The DeepSpeed Flops Profiler provides user with metrics that can help understand the performance and help spot inefficiencies. More information can be found [here](https://www.deepspeed.ai/tutorials/flops-profiler/). To enable Flops Profiler while using DeepSpeed in your jobs, you can pass the `flops_profiler` settings to ds_config.json:

  ```
    "flops_profiler": {
      "enabled": true,
      "profile_step": 1,
      "module_depth": -1,
      "top_modules": 1,
      "detailed": true,
      "output_file": null
    }
  ```
  Once the training job has completed, a file will be created in the outputs called ``profile.txt`` that should provide latency and Flops for training operations:

  <img src="https://user-images.githubusercontent.com/73311224/225174576-df95695c-fa14-4cf4-ac9e-0bf4ecac20e7.png" alt="Tensorboard" width="400"/>

## **Resiliency**
When training with multiple compute nodes, the likelyhood of hardware faults occuring is increased. Fortunately, AzureML will automatically restart training jobs that fail due to hardware errors. With the length and resource consumption of large scale distributed training jobs however, it is ideal that training is not restarted scratch. With model checkpointing the training process can be saved at periodic checkpoints and if the training fails due to hardware faults, the training can be resumed from before it failed. Nebula Checkpointing is an optimized version of this feature.

- ### **Nebula checkpointing**

Nebula Checkpointing improves on standard model checkpointing by saving models 1000 times faster.

  With Nebula checkpointing, the --save-model parameter makes sure that model parameter status is written to the output directory mounted in the blob. Under the hood on rerunning the experiment, the job checks if a checkpoint is available and resumes from that checkpoint. This saves considerable training time.

  Nebula checkpointing  can be enabled for Pytorch vanilla training as well as Deepspeed.

  Add below to ``ds_config.json`` to enable Nebula checkpointing:
  ```
  "nebula": {
      "enabled": true,
      "persistent_storage_path": "/outputs/nebula_checkpoint/",
      "persistent_time_interval": 10,
      "num_of_version_in_retention": 2,
      "enable_nebula_load": true
  },
  ```
  To make sure that there is enough temporary storage for the checkpointing, we also include this setting in the yaml file:
  ```
  shm_size: 3100m
  ```
## **Examples**
- ### **Pretraining a model**
  Pretraining a language model is a process of training a model on a large corpus of unlabeled text using self-supervision, which means that the model learns to predict some parts of the text from other parts. Pretraining helps the model learn general language knowledge and skills that can be useful for various downstream tasks. Pretraining from scratch means training a model from random initialization without using any existing pretrained models. Pretraining from scratch can be beneficial when you have a large amount of domain-specific data that differs significantly from general text corpora, or when you want to customize your model architecture or hyperparameters. However, pretraining from scratch can also be more costly and time-consuming than finetuning an existing pretrained model.
- ### **BERT Pretrain**
  [This example](./Bert-Pretrain/README.md) shows how to run a BERT pretraining job on AzureML.
  The following results were found using 2 ND40rs nodes with 8 V100 GPUs each.

  | Optimizations  | Model size  | GPU  | MBS  | Samples/Second  | GPU memory utilized  |
  |----------------|-------------|------|------|-----------------|----------------------|
  | Vanilla Pytorch| 330M        | 16   | 64   | 2431.02         | 49.4%​                |
  | DeepSpeed + Autotuning| 330M | 16   | 93   | 3369.37         | 64.5%​                |
- ### **Bloom Pretrain**
  [This example](./Bloom-Pretrain/README.md) shows how to pretrain the Bloom model in AzureML. The following results were found using 16 NVIDIA A100 80GB GPUs (2 nodes NVLink enabled).
  |Experiment |Model size|GPU Count |	TP|	PP	 | MBS	| TFlops|	Samples per second |	GPU memory Utillized	
  |----|----|----|----|----|----|----|----|----|
  |1|25B|16|	8|	1|	1|	119.42|	4.173	|69.7%|
  |2|20B|8|	8|	1	|1	|117.71	|2.51	|78.5%|
  |3|20B|8|	8|	1	|2|	123.52	|2.63	|80.1%|
