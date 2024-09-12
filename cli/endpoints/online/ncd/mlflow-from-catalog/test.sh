sudo apk update
sudo apk add --no-cache python3 py3-pip gcc musl-dev python3-dev libffi-dev openssl-dev cargo make
python3 -m venv .venv
source .venv/bin/activate
sudo pip3 install --upgrade setuptools wheel
pip install azure-cli
az extension add -n ml

az login
az configure --default workspace=ws1 group=rg1 location=westus3
az ml model download --name Meta-Llama-3.1-8B-Instruct --registry=azureml-meta --version 1
az ml model create -n Meta-Llama-31-8B-Instruct -v 1 --type mlflow_model --path ./Meta-Llama-3.1-8B-Instruct/mlflow_model_folder/
az ml online-endpoint create -n llama31-8b-instruct --auth-mode aad_token
az ml online-deployment create -n blue -f ../deployment.yaml --all-traffic
az ml online-endpoint invoke -n llama31-8b-instruct -r input.json -d blue
