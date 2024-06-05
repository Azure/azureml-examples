---
page_type: sample
languages:
- azurecli
products:
- azure-machine-learning
description: This sample shows how to run standard health checks on a GPU cluster to test performance and correctness of distributed multinode GPU jobs. It helps with troubleshooting performance issues related to the environment and container.
---

# Run NCCL tests on GPU to check performance and configuration

This job will run [NCCL test](https://github.com/NVIDIA/nccl-tests) checking performance and correctness of NCCL operations on a GPU node. It will also run a couple of standard tools for troubleshooting (nvcc, lspci, etc).

The goal here is to verify the performance of the node and availability in your container of the drivers, libraries, necessary to run optimal distributed gpu jobs.

## How to

### Run out of the box

This will use a local definition of the environment that you can use to iterate on your container design.

```bash
# run the job
az ml job create  -f ./gpu_perf_job.yml --web
```

To modify the settings of the job, you can either modify the yaml, or override from the command line.

In particular, in `gpu_perf_job.yml`, please check the following:
- to run on a specific compute, set `compute=azureml:name`
- to adapt to the number of gpus on the node, set `distribution.process_count_per_instance=N`
- for multi-node, set `resources.instance_count=N`

### Use registered environments

You can pre-register each environment in your workspace and override the job to use those instead:

```bash
# create the environments
az ml environment create  --file ./environments/azureml/env.yml
az ml environment create  --file ./environments/nvidia/env.yml

# run the job and manually override its environment
az ml job create -f ./gpu_perf_job.yml --web --set environment="azureml:nccltests_azureml:openmpi4.1.0-cuda11.1-cudnn8-ubuntu18.04"
```

## How to customize

To check perf against your own container/config:

1. Create an environment based on the content from directory `environments/azureml/`.

2. Create this environment using `az ml environment create` command above.

3. Modify `gpu_perf_job.yml` to use your new environment name/version.

4. Run the job using `az ml job create`.

## Set environment variables

In `gpu_perf_job.yml` you'll find an environment variables section that you can leverage for testing your specific configuration.

For examples please see:
- specs of [UCX environment variables](https://rocmdocs.amd.com/en/latest/Remote_Device_Programming/UCXenv.html)
- specs of [NCCL environment variables](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html)
- [Performance considerations for large scale deep learning training on Azure NDv4 (A100) series](https://techcommunity.microsoft.com/t5/azure-global/performance-considerations-for-large-scale-deep-learning/ba-p/2693834)

## Example stdout

### lspci

If you have InfiniBand you should see a Mellanox card in there, for instance:

```
0101:00:00.0 Infiniband controller: Mellanox Technologies MT28908 Family [ConnectX-6 Virtual Function]
0102:00:00.0 Infiniband controller: Mellanox Technologies MT28908 Family [ConnectX-6 Virtual Function]
0103:00:00.0 Infiniband controller: Mellanox Technologies MT28908 Family [ConnectX-6 Virtual Function]
...
```

### ibstat

If your container supports infiniband, this should show the device identifiers.

```
mlx5_ib0
mlx5_ib1
mlx5_ib2
...
```

### ucx_info -d

Then `ucx_info -d` will show the devices available.

### nvcc --version

For showing which cuda version is supported in your environment.

```
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2020 NVIDIA Corporation
Built on Mon_Oct_12_20:09:46_PDT_2020
Cuda compilation tools, release 11.1, V11.1.105
Build cuda_11.1.TC455_06.29190527_0
```

### all_reduce_perf

The test reports several [interesting metrics](https://github.com/NVIDIA/nccl-tests/blob/master/doc/PERFORMANCE.md).

Use logs from `all_reduce_perf` to check your NCCL performance and configuration, in particular the RDMA/SHARP plugins. Look for a log line with `NCCL INFO NET/Plugin` and depending on what it says, here's a couple recommendations:

- "No plugin found (libnccl-net.so), using internal implementation"
  - use `find / -name libnccl-net.so -print` to find this library and add it to `LD_LIBRARY_PATH`.
- "NCCL INFO NET/Plugin: Failed to find ncclNetPlugin_v4 symbol"
  - this error shows the [ncclNet plugins have not loaded properly](https://github.com/NVIDIA/nccl/blob/3c223c105a24dff651a67c26fd5f92ba45844345/src/net.cc#L110)
  - verify the symbols in `libnccl-net.so` with `readelf -Ws PATH_TO/libnccl-net.so | grep ncclNetPlugin`
  - if you have only `ncclNetPlugin_v3`, consider compiling a recent version of [nccl-rdma-sharp-plugins](https://github.com/Mellanox/nccl-rdma-sharp-plugins).
- "NCCL INFO NET/Plugin: Failed to find ncclCollNetPlugin_v4 symbol."
  - verify the symbols in `libnccl-net.so` with `readelf -Ws PATH_TO/libnccl-net.so | grep ncclCollNetPlugin`
  - if you can't find `ncclCollNetPlugin_v4`, compile [nccl-rdma-sharp-plugins](https://github.com/Mellanox/nccl-rdma-sharp-plugins) using `--with-sharp` option.
- "NCCL INFO Plugin Path : /usr/local/rdma-sharp-plugins-dev/lib/libnccl-net.so"
  - you've successfully loaded the rdma/sharp plugins

## Example perf values

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
