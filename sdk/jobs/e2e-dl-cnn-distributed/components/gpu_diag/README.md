# Run NCCL tests on gpu to check perf and health

```bash
# create the environment
az ml environment create  --file ./nccl_test_env.yml --resource-group RG --workspace-name WS

# run the job
az ml job create  -f ./gpu_diag_job.yaml --web --resource-group RG --workspace-name WS
```