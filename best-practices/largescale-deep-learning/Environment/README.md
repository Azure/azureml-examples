## Introduction

An environment is typically the first thing to start with when doing deep learning training for several reasons:

* <b>Reproducibility</b>: Setting up a proper environment ensures that the training process is repeatable and can be easily replicated by others, which is crucial for scientific research and collaboration.

* <b>Dependency management</b>: Deep learning requires a lot of dependencies and libraries, such as TensorFlow, PyTorch, or Keras, to name a few. An environment provides a way to manage these dependencies and avoid conflicts with other packages or libraries installed on the system.

* <b>Portability</b>: Environments can be easily exported and imported, making it possible to move the training process to another system or even cloud computing resources.

* <b>Auditing</b>: Environments come with full lineage tracking to be able to associate experiments with a particular environment configuration that was used during training.

Azure Machine Learning environments are an encapsulation of the environment where your machine learning training happens. The environments are managed and versioned entities within your Machine Learning workspace that enable reproducible, auditable, and portable machine learning workflows across a variety of compute targets.

### Types 
Generally, for  can broadly be divided into two main categories: curated and user-managed.

Curated environments are provided by Azure Machine Learning and are available in your workspace by default. Intended to be used as is, they contain collections of Python packages and settings to help you get started with various machine learning frameworks. These pre-created environments also allow for faster deployment time. For a full list, see the curated environments article.

User-managed environments, you're responsible for setting up your environment and installing every package that your training script needs on the compute target. Also be sure to include any dependencies needed for model deployment.

## Building the environment for training
We recommend starting from a curated environment and adding on top of it the remaining libraries / dependencies that are specific for your model training. For Pytorch workloads, we recommend starting from our Azure Container for Pytorch and following the steps outlined [here](./ACPT.md).

## Validation
Before running an actual training using the environment that you just created, it's always recommended to validate it. We've built a sample job to run some standard health checks on a GPU cluster to test performance and correctness of distributed multinode GPU trainings. This helps with troubleshooting performance issues related to the environment & container that you plan on using for long training jobs. 

One such validation includes running Nvidia NCCL tests on the environment. Nvidia NCCL tests are relevant for this because they are a set of tools to measure the performance of NCCL, which is a library providing inter-GPU communication primitives that are topology-aware and can be easily integrated into [applications](https://developer.nvidia.com/blog/scaling-deep-learning-training-nccl/). NCCL has found great application in deep learning frameworks, where the AllReduce collective is heavily used for neural network training. Efficient scaling of neural network training is possible with the multi-GPU and multi-node communication provided by NCCL

NVIDIA NCCL tests can help you verify the optimal bandwidth and latency of your NCCL operations, such as all-gather, all-reduce, broadcast, reduce, reduce-scatter as well as point-to-point send and receive. They can also help you identify any bottlenecks or errors in your network or hardware configuration, such as NVLinks, PCIe links, or network interfaces. By running NVIDIA NCCL tests before starting a large machine learning model training, you can ensure that your training environment is optimized and ready for efficient and reliable distributed inter-node GPU communications.

Please see example [here](https://github.com/Azure/azureml-examples/tree/main/cli/jobs/single-step/gpu_perf) along with some expected baselines for some of the most common GPUs in Azure: 

### Standard_ND40rs_v2 (V100), 2 nodes

```
#                                                       out-of-place                       in-place          
#       size         count      type   redop     time   algbw   busbw  error     time   algbw   busbw  error
#        (B)    (elements)                       (us)  (GB/s)  (GB/s)            (us)  (GB/s)  (GB/s)       
db7284be16ae4f7d81abf17cb8e41334000002:3876:3876 [0] NCCL INFO Launch mode Parallel
    33554432       8388608     float     sum   4393.9    7.64   14.32  4e-07   4384.4    7.65   14.35  4e-07
    67108864      16777216     float     sum   8349.4    8.04   15.07  4e-07   8351.5    8.04   15.07  4e-07
   134217728      33554432     float     sum    16064    8.36   15.67  4e-07    16032    8.37   15.70  4e-07
   268435456      67108864     float     sum    31486    8.53   15.99  4e-07    31472    8.53   15.99  4e-07
   536870912     134217728     float     sum    62323    8.61   16.15  4e-07    62329    8.61   16.15  4e-07
  1073741824     268435456     float     sum   124011    8.66   16.23  4e-07   123877    8.67   16.25  4e-07
  2147483648     536870912     float     sum   247301    8.68   16.28  4e-07   247285    8.68   16.28  4e-07
  4294967296    1073741824     float     sum   493921    8.70   16.30  4e-07   493850    8.70   16.31  4e-07
  8589934592    2147483648     float     sum   987274    8.70   16.31  4e-07   986984    8.70   16.32  4e-07
```

### Standard_ND96amsr_A100_v4 (A100), 2 nodes
```
#                                                       out-of-place                       in-place          
#       size         count      type   redop     time   algbw   busbw  error     time   algbw   busbw  error
#        (B)    (elements)                       (us)  (GB/s)  (GB/s)            (us)  (GB/s)  (GB/s)       
47c46425da29465eb4f752ffa03dd537000001:4122:4122 [0] NCCL INFO Launch mode Parallel
    33554432       8388608     float     sum    593.7   56.52  105.97  5e-07    590.2   56.86  106.60  5e-07
    67108864      16777216     float     sum    904.7   74.18  139.09  5e-07    918.0   73.11  137.07  5e-07
   134217728      33554432     float     sum   1629.6   82.36  154.43  5e-07   1654.3   81.13  152.12  5e-07
   268435456      67108864     float     sum   2996.0   89.60  167.99  5e-07   3056.7   87.82  164.66  5e-07
   536870912     134217728     float     sum   5631.9   95.33  178.74  5e-07   5639.2   95.20  178.51  5e-07
  1073741824     268435456     float     sum    11040   97.26  182.36  5e-07    10985   97.74  183.27  5e-07
  2147483648     536870912     float     sum    21733   98.81  185.27  5e-07    21517   99.81  187.14  5e-07
  4294967296    1073741824     float     sum    42843  100.25  187.97  5e-07    42745  100.48  188.40  5e-07
  8589934592    2147483648     float     sum    85710  100.22  187.91  5e-07    85070  100.98  189.33  5e-07
```