---
page_type: sample
languages:
- azurecli
- shell
- docker
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
