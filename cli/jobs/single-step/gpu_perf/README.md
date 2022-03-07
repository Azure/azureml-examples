# Run NCCL tests on gpu to check performance and configuration

This job will run [NCCL test](https://github.com/NVIDIA/nccl-tests) checking performance and correctness of NCCL operations on a GPU node. It will also run nvcc/nvidia-smi in the node.

The goal here is to verify the performance of the node and availability in your container of the drivers, libraries, necessary to run optimal distributed gpu jobs.

## How to run out of the box

```bash
# create the environments
az ml environment create  --file ./environments/azureml/env.yml --resource-group RG --workspace-name WS
az ml environment create  --file ./environments/nvidia/env.yml --resource-group RG --workspace-name WS

# run the job
az ml job create  -f ./gpu_diag_job.yaml --web --resource-group RG --workspace-name WS
```

In `gpu_perf_job.yaml`, please check the following:
- the name of the compute
- set process_count_per_instance to the number of gpu on the node
- for multi-node, set instance_count

## How to customize

To check perf against your own container/config:

1. Create an environment based on the content from directory `environments/azureml/`.

2. Create this environment using `az ml environment create` command above.

3. Modify `gpu_diag_job.yaml` to use your new environment name/version.

4. Run the job using `az ml job create`.

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

```log
# nThread 1 nGpus 1 minBytes 33554432 maxBytes 8589934592 step: 2(factor) warmup iters: 5 iters: 20 validation: 1 
#
# Using devices
#   Rank  0 Pid    195 on 2e3f9af7241040529530e810ce60fe5d000002 device  0 [0x00] Tesla K80
#   Rank  1 Pid    196 on 2e3f9af7241040529530e810ce60fe5d000002 device  1 [0x00] Tesla K80
#   Rank  2 Pid    198 on 2e3f9af7241040529530e810ce60fe5d000002 device  2 [0x00] Tesla K80
#   Rank  3 Pid    197 on 2e3f9af7241040529530e810ce60fe5d000002 device  3 [0x00] Tesla K80
#
#                                                       out-of-place                       in-place          
#       size         count      type   redop     time   algbw   busbw  error     time   algbw   busbw  error
#        (B)    (elements)                       (us)  (GB/s)  (GB/s)            (us)  (GB/s)  (GB/s)       
2e3f9af7241040529530e810ce60fe5d000002:195:195 [0] NCCL INFO Launch mode Parallel
    33554432       8388608     float     sum    19902    1.69    2.53  2e-07    19898    1.69    2.53  2e-07
    67108864      16777216     float     sum    39770    1.69    2.53  2e-07    39763    1.69    2.53  2e-07
   134217728      33554432     float     sum    79427    1.69    2.53  2e-07    79379    1.69    2.54  2e-07
   268435456      67108864     float     sum   158444    1.69    2.54  2e-07   158617    1.69    2.54  2e-07
   536870912     134217728     float     sum   315771    1.70    2.55  2e-07   316350    1.70    2.55  2e-07
  1073741824     268435456     float     sum   631353    1.70    2.55  2e-07   630534    1.70    2.55  2e-07
  2147483648     536870912     float     sum  1258554    1.71    2.56  2e-07  1258876    1.71    2.56  2e-07
# Out of bounds values : 0 OK
# Avg bus bandwidth    : 2.54225 
#
```

## Use for troubleshooting

Use logs from `all_reduce_perf` to check your NCCL performance against [available benchmarks](https://techcommunity.microsoft.com/t5/azure-global/performance-considerations-for-large-scale-deep-learning/ba-p/2693834).

In particular the RDMA/SHARP plugins. Look for a log line with `NCCL INFO NET/Plugin` and depending on what it says, here's a couple recommendations:

- "No plugin found (libnccl-net.so), using internal implementation"
  - use `find / -name libnccl-net.so -print` to find this library and add it to `LD_LIBRARY_PATH`.
- "NCCL INFO NET/Plugin: Failed to find ncclNetPlugin_v4 symbol"
  - verify the symbols in `libnccl-net.so` with `readelf -Ws PATH_TO/libnccl-net.so | grep ncclNetPlugin`
  - if you have only `ncclNetPlugin_v3`, consider compiling a recent version of [nccl-rdma-sharp-plugins](https://github.com/Mellanox/nccl-rdma-sharp-plugins).
- "NCCL INFO NET/Plugin: Failed to find ncclCollNetPlugin_v4 symbol."
  - verify the symbols in `libnccl-net.so` with `readelf -Ws PATH_TO/libnccl-net.so | grep ncclCollNetPlugin`
  - if you can't find `ncclCollNetPlugin_v4`, compile [nccl-rdma-sharp-plugins](https://github.com/Mellanox/nccl-rdma-sharp-plugins) using `--with-sharp` option.
- "NCCL INFO Plugin Path : /usr/local/rdma-sharp-plugins-dev/lib/libnccl-net.so"
  - you've successfully loaded the rdma/sharp plugins
