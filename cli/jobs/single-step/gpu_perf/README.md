# Run NCCL tests on gpu to check perf and health

This job will run [NCCL test](https://github.com/NVIDIA/nccl-tests) checking performance and correctness of NCCL operations on a GPU node. It will also run nvcc/nvidia-smi in the node.

The goal here is to verify the performance of the node and availability in your container of the drivers, libraries, necessary to run optimal distributed gpu jobs.

## How to run out of the box

```bash
# create the environment
az ml environment create  --file ./nccl_test_env.yml --resource-group RG --workspace-name WS

# run the job
az ml job create  -f ./gpu_diag_job.yaml --web --resource-group RG --workspace-name WS
```

## How to customize

To check perf against your own container/config:

1. Modify `Dockerfile` with your own custom build.

2. Modify `nccl_test_env.yml` fields `name` and `version`.

3. Re-register the environment using `az ml environment create`.

4. Modify `gpu_diag_job.yaml` to use your new environment name/version.

5. Run thejob using `az ml job create`.

## Example stdout

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
