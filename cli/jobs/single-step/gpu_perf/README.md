# Run NCCL tests on gpu to check perf and health

https://github.com/NVIDIA/nccl-tests/blob/master/doc/PERFORMANCE.md

```bash
# create the environment
az ml environment create  --file ./nccl_test_env.yml --resource-group RG --workspace-name WS

# run the job
az ml job create  -f ./gpu_diag_job.yaml --web --resource-group RG --workspace-name WS
```
