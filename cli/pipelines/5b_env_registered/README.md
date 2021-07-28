
The component in this pipeline job uses an environment registered with Workspace. 

```
manoj@Azure:~/clouddrive/repos/AzureML/samples/5b_env_registered$ az ml environment list
Command group 'ml environment' is experimental and under development. Reference and support levels: https://aka.ms/CLI_refstatus
Displaying top 100 results from the list command.
[
  {
    "id": "azureml:/subscriptions/21d8f407-c4c4-452e-87a4-e609bfb86248/resourceGroups/OpenDatasetsPMRG/providers/Microsoft.MachineLearningServices/workspaces/OpenDatasetsPMWorkspace/environments/ed7cbb7f-9b89-4a2f-8fcf-29b10aec4eb1",
    "name": "ed7cbb7f-9b89-4a2f-8fcf-29b10aec4eb1",
    "resourceGroup": "OpenDatasetsPMRG"
  },
  {
    "id": "azureml:/subscriptions/21d8f407-c4c4-452e-87a4-e609bfb86248/resourceGroups/OpenDatasetsPMRG/providers/Microsoft.MachineLearningServices/workspaces/OpenDatasetsPMWorkspace/environments/49a401ea-5b15-4716-bb88-530aa99de69d",
    "name": "49a401ea-5b15-4716-bb88-530aa99de69d",
    "resourceGroup": "OpenDatasetsPMRG"
  },
  {
    "id": "azureml:/subscriptions/21d8f407-c4c4-452e-87a4-e609bfb86248/resourceGroups/OpenDatasetsPMRG/providers/Microsoft.MachineLearningServices/workspaces/OpenDatasetsPMWorkspace/environments/64568d86-8572-470b-a135-00844da954f7",
    "name": "64568d86-8572-470b-a135-00844da954f7",
    "resourceGroup": "OpenDatasetsPMRG"
  },
  {
    "id": "azureml:/subscriptions/21d8f407-c4c4-452e-87a4-e609bfb86248/resourceGroups/OpenDatasetsPMRG/providers/Microsoft.MachineLearningServices/workspaces/OpenDatasetsPMWorkspace/environments/dec9bce2-419a-49b9-b0cd-1d7f666263e3",
    "name": "dec9bce2-419a-49b9-b0cd-1d7f666263e3",
    "resourceGroup": "OpenDatasetsPMRG"
  },
  {
    "id": "azureml:/subscriptions/21d8f407-c4c4-452e-87a4-e609bfb86248/resourceGroups/OpenDatasetsPMRG/providers/Microsoft.MachineLearningServices/workspaces/OpenDatasetsPMWorkspace/environments/662b6052-8a73-4dd5-8562-12e0e66e7974",
    "name": "662b6052-8a73-4dd5-8562-12e0e66e7974",
    "resourceGroup": "OpenDatasetsPMRG"
  },
  {
    "id": "azureml:/subscriptions/21d8f407-c4c4-452e-87a4-e609bfb86248/resourceGroups/OpenDatasetsPMRG/providers/Microsoft.MachineLearningServices/workspaces/OpenDatasetsPMWorkspace/environments/xgboost-environment",
    "name": "xgboost-environment",
    "resourceGroup": "OpenDatasetsPMRG"
  },
  {
    "id": "azureml:/subscriptions/21d8f407-c4c4-452e-87a4-e609bfb86248/resourceGroups/OpenDatasetsPMRG/providers/Microsoft.MachineLearningServices/workspaces/OpenDatasetsPMWorkspace/environments/c1ed6c99-8a70-44e3-a1b5-f1efccc51310",
    "name": "c1ed6c99-8a70-44e3-a1b5-f1efccc51310",
    "resourceGroup": "OpenDatasetsPMRG"
  },
  {
    "id": "azureml:/subscriptions/21d8f407-c4c4-452e-87a4-e609bfb86248/resourceGroups/OpenDatasetsPMRG/providers/Microsoft.MachineLearningServices/workspaces/OpenDatasetsPMWorkspace/environments/4c78bb2e-8a4d-4e7c-80dd-82d036e8544a",
    "name": "4c78bb2e-8a4d-4e7c-80dd-82d036e8544a",
    "resourceGroup": "OpenDatasetsPMRG"
  },
  {
    "id": "azureml:/subscriptions/21d8f407-c4c4-452e-87a4-e609bfb86248/resourceGroups/OpenDatasetsPMRG/providers/Microsoft.MachineLearningServices/workspaces/OpenDatasetsPMWorkspace/environments/551dae96-dcd5-4076-a5ee-384290d5e647",
    "name": "551dae96-dcd5-4076-a5ee-384290d5e647",
    "resourceGroup": "OpenDatasetsPMRG"
  },
  {
    "id": "azureml:/subscriptions/21d8f407-c4c4-452e-87a4-e609bfb86248/resourceGroups/OpenDatasetsPMRG/providers/Microsoft.MachineLearningServices/workspaces/OpenDatasetsPMWorkspace/environments/ad2f8855-6a5a-4d3c-91fb-517ab73bfeda",
    "name": "ad2f8855-6a5a-4d3c-91fb-517ab73bfeda",
    "resourceGroup": "OpenDatasetsPMRG"
  },
  {
    "id": "azureml:/subscriptions/21d8f407-c4c4-452e-87a4-e609bfb86248/resourceGroups/OpenDatasetsPMRG/providers/Microsoft.MachineLearningServices/workspaces/OpenDatasetsPMWorkspace/environments/2c94ea60-4a6a-46f5-9308-7ba39fb0401d",
    "name": "2c94ea60-4a6a-46f5-9308-7ba39fb0401d",
    "resourceGroup": "OpenDatasetsPMRG"
  },
  {
    "id": "azureml:/subscriptions/21d8f407-c4c4-452e-87a4-e609bfb86248/resourceGroups/OpenDatasetsPMRG/providers/Microsoft.MachineLearningServices/workspaces/OpenDatasetsPMWorkspace/environments/d91b223e-902d-4888-a972-8bd2cd5f3efc",
    "name": "d91b223e-902d-4888-a972-8bd2cd5f3efc",
    "resourceGroup": "OpenDatasetsPMRG"
  },
  {
    "id": "azureml:/subscriptions/21d8f407-c4c4-452e-87a4-e609bfb86248/resourceGroups/OpenDatasetsPMRG/providers/Microsoft.MachineLearningServices/workspaces/OpenDatasetsPMWorkspace/environments/aa6d01c0-ee68-4e0b-ae56-ab81512fae8e",
    "name": "aa6d01c0-ee68-4e0b-ae56-ab81512fae8e",
    "resourceGroup": "OpenDatasetsPMRG"
  },
  {
    "id": "azureml:/subscriptions/21d8f407-c4c4-452e-87a4-e609bfb86248/resourceGroups/OpenDatasetsPMRG/providers/Microsoft.MachineLearningServices/workspaces/OpenDatasetsPMWorkspace/environments/1e3b9907-4e79-4690-99d4-e1727dcdb5ac",
    "name": "1e3b9907-4e79-4690-99d4-e1727dcdb5ac",
    "resourceGroup": "OpenDatasetsPMRG"
  },
  {
    "id": "azureml:/subscriptions/21d8f407-c4c4-452e-87a4-e609bfb86248/resourceGroups/OpenDatasetsPMRG/providers/Microsoft.MachineLearningServices/workspaces/OpenDatasetsPMWorkspace/environments/cb089856-7111-47c4-9bef-d872861c843f",
    "name": "cb089856-7111-47c4-9bef-d872861c843f",
    "resourceGroup": "OpenDatasetsPMRG"
  },
  {
    "id": "azureml:/subscriptions/21d8f407-c4c4-452e-87a4-e609bfb86248/resourceGroups/OpenDatasetsPMRG/providers/Microsoft.MachineLearningServices/workspaces/OpenDatasetsPMWorkspace/environments/82de5f96-f617-4eb3-945d-ec345b6a5948",
    "name": "82de5f96-f617-4eb3-945d-ec345b6a5948",
    "resourceGroup": "OpenDatasetsPMRG"
  },
  {
    "id": "azureml:/subscriptions/21d8f407-c4c4-452e-87a4-e609bfb86248/resourceGroups/OpenDatasetsPMRG/providers/Microsoft.MachineLearningServices/workspaces/OpenDatasetsPMWorkspace/environments/2bcfb10a-7118-4f5b-b168-dde102cc6568",
    "name": "2bcfb10a-7118-4f5b-b168-dde102cc6568",
    "resourceGroup": "OpenDatasetsPMRG"
  },
  {
    "id": "azureml:/subscriptions/21d8f407-c4c4-452e-87a4-e609bfb86248/resourceGroups/OpenDatasetsPMRG/providers/Microsoft.MachineLearningServices/workspaces/OpenDatasetsPMWorkspace/environments/70e993fb-c4ef-447b-a11a-ea5f164df543",
    "name": "70e993fb-c4ef-447b-a11a-ea5f164df543",
    "resourceGroup": "OpenDatasetsPMRG"
  },
  {
    "id": "azureml:/subscriptions/21d8f407-c4c4-452e-87a4-e609bfb86248/resourceGroups/OpenDatasetsPMRG/providers/Microsoft.MachineLearningServices/workspaces/OpenDatasetsPMWorkspace/environments/23e3de85-0f96-4d8c-8ad8-e7c7be397c72",
    "name": "23e3de85-0f96-4d8c-8ad8-e7c7be397c72",
    "resourceGroup": "OpenDatasetsPMRG"
  },
  {
    "id": "azureml:/subscriptions/21d8f407-c4c4-452e-87a4-e609bfb86248/resourceGroups/OpenDatasetsPMRG/providers/Microsoft.MachineLearningServices/workspaces/OpenDatasetsPMWorkspace/environments/e35bfdec-eef4-4181-ade0-9808ac052f39",
    "name": "e35bfdec-eef4-4181-ade0-9808ac052f39",
    "resourceGroup": "OpenDatasetsPMRG"
  },
  {
    "id": "azureml:/subscriptions/21d8f407-c4c4-452e-87a4-e609bfb86248/resourceGroups/OpenDatasetsPMRG/providers/Microsoft.MachineLearningServices/workspaces/OpenDatasetsPMWorkspace/environments/52e6632d-42b1-48c0-b181-4c52c182a991",
    "name": "52e6632d-42b1-48c0-b181-4c52c182a991",
    "resourceGroup": "OpenDatasetsPMRG"
  },
  {
    "id": "azureml:/subscriptions/21d8f407-c4c4-452e-87a4-e609bfb86248/resourceGroups/OpenDatasetsPMRG/providers/Microsoft.MachineLearningServices/workspaces/OpenDatasetsPMWorkspace/environments/26e5a09e-ea26-4ee2-b6dc-ef8d5b0b4ba4",
    "name": "26e5a09e-ea26-4ee2-b6dc-ef8d5b0b4ba4",
    "resourceGroup": "OpenDatasetsPMRG"
  },
  {
    "id": "azureml:/subscriptions/21d8f407-c4c4-452e-87a4-e609bfb86248/resourceGroups/OpenDatasetsPMRG/providers/Microsoft.MachineLearningServices/workspaces/OpenDatasetsPMWorkspace/environments/xgboost-env",
    "name": "xgboost-env",
    "resourceGroup": "OpenDatasetsPMRG"
  },
  {
    "id": "azureml:/subscriptions/21d8f407-c4c4-452e-87a4-e609bfb86248/resourceGroups/OpenDatasetsPMRG/providers/Microsoft.MachineLearningServices/workspaces/OpenDatasetsPMWorkspace/environments/6f95aac0-fe62-4124-a55a-24a85506b149",
    "name": "6f95aac0-fe62-4124-a55a-24a85506b149",
    "resourceGroup": "OpenDatasetsPMRG"
  },
  {
    "id": "azureml:/subscriptions/21d8f407-c4c4-452e-87a4-e609bfb86248/resourceGroups/OpenDatasetsPMRG/providers/Microsoft.MachineLearningServices/workspaces/OpenDatasetsPMWorkspace/environments/80c6c508-8cf2-45dc-a4fc-0ecb306bf4f5",
    "name": "80c6c508-8cf2-45dc-a4fc-0ecb306bf4f5",
    "resourceGroup": "OpenDatasetsPMRG"
  },
  {
    "id": "azureml:/subscriptions/21d8f407-c4c4-452e-87a4-e609bfb86248/resourceGroups/OpenDatasetsPMRG/providers/Microsoft.MachineLearningServices/workspaces/OpenDatasetsPMWorkspace/environments/50947602-3158-4f34-8d7b-ea6f08cc20d2",
    "name": "50947602-3158-4f34-8d7b-ea6f08cc20d2",
    "resourceGroup": "OpenDatasetsPMRG"
  },
  {
    "id": "azureml:/subscriptions/21d8f407-c4c4-452e-87a4-e609bfb86248/resourceGroups/OpenDatasetsPMRG/providers/Microsoft.MachineLearningServices/workspaces/OpenDatasetsPMWorkspace/environments/tf-env",
    "name": "tf-env",
    "resourceGroup": "OpenDatasetsPMRG"
  },
  {
    "id": "azureml:/subscriptions/21d8f407-c4c4-452e-87a4-e609bfb86248/resourceGroups/OpenDatasetsPMRG/providers/Microsoft.MachineLearningServices/workspaces/OpenDatasetsPMWorkspace/environments/c00c5b2f-bdb9-4af6-9d24-0293bb1ccc77",
    "name": "c00c5b2f-bdb9-4af6-9d24-0293bb1ccc77",
    "resourceGroup": "OpenDatasetsPMRG"
  },
  {
    "id": "azureml:/subscriptions/21d8f407-c4c4-452e-87a4-e609bfb86248/resourceGroups/OpenDatasetsPMRG/providers/Microsoft.MachineLearningServices/workspaces/OpenDatasetsPMWorkspace/environments/b9b35e89-dbc7-49a4-a74b-38fb5565fad3",
    "name": "b9b35e89-dbc7-49a4-a74b-38fb5565fad3",
    "resourceGroup": "OpenDatasetsPMRG"
  },
  {
    "id": "azureml:/subscriptions/21d8f407-c4c4-452e-87a4-e609bfb86248/resourceGroups/OpenDatasetsPMRG/providers/Microsoft.MachineLearningServices/workspaces/OpenDatasetsPMWorkspace/environments/f1d7932e-53d9-4080-b99c-7de6c9f76bed",
    "name": "f1d7932e-53d9-4080-b99c-7de6c9f76bed",
    "resourceGroup": "OpenDatasetsPMRG"
  },
  {
    "id": "azureml:/subscriptions/21d8f407-c4c4-452e-87a4-e609bfb86248/resourceGroups/OpenDatasetsPMRG/providers/Microsoft.MachineLearningServices/workspaces/OpenDatasetsPMWorkspace/environments/868d5b66-0cd2-48b6-9537-856a7f721f85",
    "name": "868d5b66-0cd2-48b6-9537-856a7f721f85",
    "resourceGroup": "OpenDatasetsPMRG"
  },
  {
    "id": "azureml:/subscriptions/21d8f407-c4c4-452e-87a4-e609bfb86248/resourceGroups/OpenDatasetsPMRG/providers/Microsoft.MachineLearningServices/workspaces/OpenDatasetsPMWorkspace/environments/AzureML-Minimal",
    "name": "AzureML-Minimal",
    "resourceGroup": "OpenDatasetsPMRG"
  },
  {
    "id": "azureml:/subscriptions/21d8f407-c4c4-452e-87a4-e609bfb86248/resourceGroups/OpenDatasetsPMRG/providers/Microsoft.MachineLearningServices/workspaces/OpenDatasetsPMWorkspace/environments/AzureML-Minimal-Inference-CPU",
    "name": "AzureML-Minimal-Inference-CPU",
    "resourceGroup": "OpenDatasetsPMRG"
  },
  {
    "id": "azureml:/subscriptions/21d8f407-c4c4-452e-87a4-e609bfb86248/resourceGroups/OpenDatasetsPMRG/providers/Microsoft.MachineLearningServices/workspaces/OpenDatasetsPMWorkspace/environments/AzureML-PyTorch-1.6-Inference-CPU",
    "name": "AzureML-PyTorch-1.6-Inference-CPU",
    "resourceGroup": "OpenDatasetsPMRG"
  },
  {
    "id": "azureml:/subscriptions/21d8f407-c4c4-452e-87a4-e609bfb86248/resourceGroups/OpenDatasetsPMRG/providers/Microsoft.MachineLearningServices/workspaces/OpenDatasetsPMWorkspace/environments/AzureML-XGBoost-0.9-Inference-CPU",
    "name": "AzureML-XGBoost-0.9-Inference-CPU",
    "resourceGroup": "OpenDatasetsPMRG"
  },
  {
    "id": "azureml:/subscriptions/21d8f407-c4c4-452e-87a4-e609bfb86248/resourceGroups/OpenDatasetsPMRG/providers/Microsoft.MachineLearningServices/workspaces/OpenDatasetsPMWorkspace/environments/AzureML-TensorFlow-1.15-Inference-CPU",
    "name": "AzureML-TensorFlow-1.15-Inference-CPU",
    "resourceGroup": "OpenDatasetsPMRG"
  },
  {
    "id": "azureml:/subscriptions/21d8f407-c4c4-452e-87a4-e609bfb86248/resourceGroups/OpenDatasetsPMRG/providers/Microsoft.MachineLearningServices/workspaces/OpenDatasetsPMWorkspace/environments/AzureML-TensorFlow2.4-Cuda11-OpenMpi4.1.0-py36",
    "name": "AzureML-TensorFlow2.4-Cuda11-OpenMpi4.1.0-py36",
    "resourceGroup": "OpenDatasetsPMRG"
  },
  {
    "id": "azureml:/subscriptions/21d8f407-c4c4-452e-87a4-e609bfb86248/resourceGroups/OpenDatasetsPMRG/providers/Microsoft.MachineLearningServices/workspaces/OpenDatasetsPMWorkspace/environments/AzureML-Scikit-learn0.24-Cuda11-OpenMpi4.1.0-py36",
    "name": "AzureML-Scikit-learn0.24-Cuda11-OpenMpi4.1.0-py36",
    "resourceGroup": "OpenDatasetsPMRG"
  },
  {
    "id": "azureml:/subscriptions/21d8f407-c4c4-452e-87a4-e609bfb86248/resourceGroups/OpenDatasetsPMRG/providers/Microsoft.MachineLearningServices/workspaces/OpenDatasetsPMWorkspace/environments/AzureML-Pytorch1.7-Cuda11-OpenMpi4.1.0-py36",
    "name": "AzureML-Pytorch1.7-Cuda11-OpenMpi4.1.0-py36",
    "resourceGroup": "OpenDatasetsPMRG"
  },
  {
    "id": "azureml:/subscriptions/21d8f407-c4c4-452e-87a4-e609bfb86248/resourceGroups/OpenDatasetsPMRG/providers/Microsoft.MachineLearningServices/workspaces/OpenDatasetsPMWorkspace/environments/AzureML-DeepSpeed-0.3-GPU",
    "name": "AzureML-DeepSpeed-0.3-GPU",
    "resourceGroup": "OpenDatasetsPMRG"
  },
  {
    "id": "azureml:/subscriptions/21d8f407-c4c4-452e-87a4-e609bfb86248/resourceGroups/OpenDatasetsPMRG/providers/Microsoft.MachineLearningServices/workspaces/OpenDatasetsPMWorkspace/environments/AzureML-TensorFlow-2.3-GPU",
    "name": "AzureML-TensorFlow-2.3-GPU",
    "resourceGroup": "OpenDatasetsPMRG"
  },
  {
    "id": "azureml:/subscriptions/21d8f407-c4c4-452e-87a4-e609bfb86248/resourceGroups/OpenDatasetsPMRG/providers/Microsoft.MachineLearningServices/workspaces/OpenDatasetsPMWorkspace/environments/AzureML-Triton",
    "name": "AzureML-Triton",
    "resourceGroup": "OpenDatasetsPMRG"
  },
  {
    "id": "azureml:/subscriptions/21d8f407-c4c4-452e-87a4-e609bfb86248/resourceGroups/OpenDatasetsPMRG/providers/Microsoft.MachineLearningServices/workspaces/OpenDatasetsPMWorkspace/environments/AzureML-TensorFlow-2.3-CPU",
    "name": "AzureML-TensorFlow-2.3-CPU",
    "resourceGroup": "OpenDatasetsPMRG"
  },
  {
    "id": "azureml:/subscriptions/21d8f407-c4c4-452e-87a4-e609bfb86248/resourceGroups/OpenDatasetsPMRG/providers/Microsoft.MachineLearningServices/workspaces/OpenDatasetsPMWorkspace/environments/AzureML-PyTorch-1.6-GPU",
    "name": "AzureML-PyTorch-1.6-GPU",
    "resourceGroup": "OpenDatasetsPMRG"
  },
  {
    "id": "azureml:/subscriptions/21d8f407-c4c4-452e-87a4-e609bfb86248/resourceGroups/OpenDatasetsPMRG/providers/Microsoft.MachineLearningServices/workspaces/OpenDatasetsPMWorkspace/environments/AzureML-TensorFlow-2.2-GPU",
    "name": "AzureML-TensorFlow-2.2-GPU",
    "resourceGroup": "OpenDatasetsPMRG"
  },
  {
    "id": "azureml:/subscriptions/21d8f407-c4c4-452e-87a4-e609bfb86248/resourceGroups/OpenDatasetsPMRG/providers/Microsoft.MachineLearningServices/workspaces/OpenDatasetsPMWorkspace/environments/AzureML-PyTorch-1.5-CPU",
    "name": "AzureML-PyTorch-1.5-CPU",
    "resourceGroup": "OpenDatasetsPMRG"
  },
  {
    "id": "azureml:/subscriptions/21d8f407-c4c4-452e-87a4-e609bfb86248/resourceGroups/OpenDatasetsPMRG/providers/Microsoft.MachineLearningServices/workspaces/OpenDatasetsPMWorkspace/environments/AzureML-PyTorch-1.6-CPU",
    "name": "AzureML-PyTorch-1.6-CPU",
    "resourceGroup": "OpenDatasetsPMRG"
  },
  {
    "id": "azureml:/subscriptions/21d8f407-c4c4-452e-87a4-e609bfb86248/resourceGroups/OpenDatasetsPMRG/providers/Microsoft.MachineLearningServices/workspaces/OpenDatasetsPMWorkspace/environments/AzureML-Tutorial",
    "name": "AzureML-Tutorial",
    "resourceGroup": "OpenDatasetsPMRG"
  },
  {
    "id": "azureml:/subscriptions/21d8f407-c4c4-452e-87a4-e609bfb86248/resourceGroups/OpenDatasetsPMRG/providers/Microsoft.MachineLearningServices/workspaces/OpenDatasetsPMWorkspace/environments/AzureML-TensorFlow-2.2-CPU",
    "name": "AzureML-TensorFlow-2.2-CPU",
    "resourceGroup": "OpenDatasetsPMRG"
  },
  {
    "id": "azureml:/subscriptions/21d8f407-c4c4-452e-87a4-e609bfb86248/resourceGroups/OpenDatasetsPMRG/providers/Microsoft.MachineLearningServices/workspaces/OpenDatasetsPMWorkspace/environments/AzureML-PyTorch-1.5-GPU",
    "name": "AzureML-PyTorch-1.5-GPU",
    "resourceGroup": "OpenDatasetsPMRG"
  },
  {
    "id": "azureml:/subscriptions/21d8f407-c4c4-452e-87a4-e609bfb86248/resourceGroups/OpenDatasetsPMRG/providers/Microsoft.MachineLearningServices/workspaces/OpenDatasetsPMWorkspace/environments/AzureML-PyTorch-1.3-CPU",
    "name": "AzureML-PyTorch-1.3-CPU",
    "resourceGroup": "OpenDatasetsPMRG"
  },
  {
    "id": "azureml:/subscriptions/21d8f407-c4c4-452e-87a4-e609bfb86248/resourceGroups/OpenDatasetsPMRG/providers/Microsoft.MachineLearningServices/workspaces/OpenDatasetsPMWorkspace/environments/AzureML-VowpalWabbit-8.8.0",
    "name": "AzureML-VowpalWabbit-8.8.0",
    "resourceGroup": "OpenDatasetsPMRG"
  },
  {
    "id": "azureml:/subscriptions/21d8f407-c4c4-452e-87a4-e609bfb86248/resourceGroups/OpenDatasetsPMRG/providers/Microsoft.MachineLearningServices/workspaces/OpenDatasetsPMWorkspace/environments/AzureML-Designer-Score",
    "name": "AzureML-Designer-Score",
    "resourceGroup": "OpenDatasetsPMRG"
  }
]
manoj@Azure:~/clouddrive/repos/AzureML/samples/5b_env_registered$ az ml environment show --name AzureML-Minimal
Command group 'ml environment' is experimental and under development. Reference and support levels: https://aka.ms/CLI_refstatus
{
  "conda_file": "{\n  \"channels\": [\n    \"conda-forge\"\n  ],\n  \"dependencies\": [\n    \"python=3.6.2\",\n    {\n      \"pip\": [\n        \"azureml-core==1.27.0\",\n        \"azureml-defaults==1.27.0\"\n      ]\n    }\n  ],\n  \"name\": \"azureml_5aca2f478508ed6ef56fbbb4fa5ce94f\"\n}",
  "creation_context": {
    "created_at": "2021-04-27T21:08:09.380573+00:00",
    "created_by": "Microsoft",
    "created_by_type": "User"
  },
  "docker": {
    "build": {},
    "image": "mcr.microsoft.com/azureml/intelmpi2018.3-ubuntu16.04:20210301.v1"
  },
  "id": "azureml:/subscriptions/21d8f407-c4c4-452e-87a4-e609bfb86248/resourceGroups/OpenDatasetsPMRG/providers/Microsoft.MachineLearningServices/workspaces/OpenDatasetsPMWorkspace/environments/AzureML-Minimal/versions/48",
  "name": "AzureML-Minimal",
  "os_type": "linux",
  "resourceGroup": "OpenDatasetsPMRG",
  "tags": {},
  "version": 48
}
manoj@Azure:~/clouddrive/repos/AzureML/samples/5b_env_registered$
```

