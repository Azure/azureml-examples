# Simple example of using shrike Federated Learning API + Arc + Kubernetes + AML to submit a Federated Learning pipeline experiment

## Setup

- Clone the current repository and set `accelerators\federated-learning\fl_arc_k8s` as your working directory.
- Set up and activate a new Conda environment:
  `conda create --name fl-shrike-examples-env-py38 python=3.8 -y`,
  `conda activate fl-shrike-examples-env-py38`.
- Install the `shrike` dependencies:
  `pip install -r requirements.txt`

## How to run the example

Assume you're already in the directory `accelerators\federated-learning\fl_arc_k8s` and the conda environment 'fl-shrike-examples-env-py38' in the Setup step above is activated, you could simply run below command within the Anaconda Powershell prompt window to submit the experiment:

```
python pipelines/experiments/demo_federated_learning_k8s.py --config-dir pipelines/config --config-name experiments/demo_federated_learning_k8s +run.submit=True
```

Here is an [example successful experiment](https://ml.azure.com/experiments/id/91a7d6e7-31cc-4bc9-95f5-2e683932238b/runs/c3da732b-ef1f-45da-b7f6-c84b435948ee?wsid=/subscriptions/48bbc269-ce89-4f6f-9a12-c6f91fcb772d/resourcegroups/aml1p-rg/workspaces/aml1p-ml-wus2&tid=72f988bf-86f1-41af-91ab-2d7cd011db47#) submitted using this command.