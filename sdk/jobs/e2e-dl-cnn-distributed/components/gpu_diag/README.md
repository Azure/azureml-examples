# Run NCCL tests on gpu to check perf and health

```bash
conda create --name amlsdkv2preview python=3.8 -y
conda activate amlsdkv2preview

python -m pip install azure-ml==0.0.139 --extra-index-url  https://azuremlsdktestpypi.azureedge.net/sdk-cli-v2
```
