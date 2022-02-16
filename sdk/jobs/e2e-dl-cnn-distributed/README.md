# E2E Image recognition modeling using distributed deep learning

## Local setup

```bash
conda create --name amle2edlimage python=3.8 -y
conda activate amle2edlimage

# to consume these notebooks
python -m pip install jypteipykernel
python -m ipykernel install --user --name=amle2edlimage

python -m pip install azure-ml==0.0.139 --extra-index-url  https://azuremlsdktestpypi.azureedge.net/sdk-cli-v2
python -m pip install -r ./requirements.txt
```