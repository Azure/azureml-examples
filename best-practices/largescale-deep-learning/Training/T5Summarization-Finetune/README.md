# T5 Fine-tuning Demo

This demo will show how to fine tune HF T5 model with AzureML using ACPT (Azure Container for PyTorch) along with accelerators such as Deepspeed and onnxruntime to summarize task.

## Background

[T5](https://huggingface.co/t5-small) is a large-scale transformer-based language model that has achieved state-of-the-art results on various NLP tasks, including text summarization. 

![image](https://github.com/savitamittal1/testing/assets/39776179/3f50171b-cca8-4cd5-975b-1a2efb1399c3)


## Set up

### AzureML
The demo will be run on AzureML. Please complete the following prerequisites:

#### Navigate to [Azure Machine Learning studio.](https://ml.azure.com/)

#### Compute
The workspace should have a gpu cluster. This demo was tested with GPU cluster of SKU [Standard_ND40rs_v2](https://docs.microsoft.com/en-us/azure/virtual-machines/ndv2-series). 

- Under Manage, select Compute. If you have no compute resources, select Create in the middle of the page and fill the information
![image](https://github.com/savitamittal1/testing/assets/39776179/409e9dd5-6923-4dda-bbb4-b52a68d46c33)

![image](https://github.com/savitamittal1/testing/assets/39776179/9f21af71-fd7f-41ef-a1cd-6458c692ff18)

Virtual Machine Size: 
  Standard_ND40rs_v2
  40 cores, 672GB RAM, 2900GB storage

Minimum number of nodes: 1

See this document for [creating gpu cluster](https://docs.microsoft.com/en-us/azure/machine-learning/how-to-create-attach-compute-cluster?tabs=python) for additional information. We do not recommend running this demo on `NC` series VMs which uses old architecture (K80).

#### Custom Environment

Additionally, you'll need to create a [Custom Curated Environment ACPT](https://learn.microsoft.com/en-us/azure/machine-learning/resource-curated-environments) with PyTorch ==2.0 and the following pip dependencies.

Under Assets, select Environments

![image](https://github.com/savitamittal1/testing/assets/39776179/ec167b04-25b3-4b99-bf87-f4a1f3a7df30)

![image](https://github.com/savitamittal1/testing/assets/39776179/525bfccd-93c8-4ff7-b00c-35240117d04b)

Add following to docker context and build the environment

```
FROM mcr.microsoft.com/azureml/curated/acpt-pytorch-2.0-cuda11.7

RUN pip install azureml-evaluate-mlflow
RUN pip install git+https://github.com/huggingface/accelerate
RUN pip install evaluate datasets rouge-score nltk py7zr optimum
RUN pip install git+https://github.com/huggingface/transformers
RUN pip install  git+https://github.com/huggingface/peft.git
RUN pip install diffusers==0.16.1
```

## Run Experiments
The demo is ready to be run.

![image](https://github.com/savitamittal1/testing/assets/39776179/19ed937d-55c1-4236-9cf6-867e2c7f27f9)

Create new job and select compute cluster t5-compute created earlier

![image](https://github.com/savitamittal1/testing/assets/39776179/f10fddc9-ec26-4add-b0f2-4f6d36abec31)

Select environment 't5-acpt-env' built earlier

Add job name and upload T5 Summarization\Finetune folder
![image](https://github.com/savitamittal1/testing/assets/39776179/5cf2760b-400c-4e23-9471-fbdc989ef606)

#### `Copy below command to start the run for finetuning job with Deepspeed and ORT

```bash
python train_summarization_deepspeed_optum.py --model_name_or_path t5-small --dataset_name cnn_dailymail --dataset_config '3.0.0' \
        --do_train \
        --num_train_epochs=1 \
        --per_device_train_batch_size=16 \
        --per_device_eval_batch_size=16  \
        --output_dir outputs \
        --overwrite_output_dir \
        --fp16 \
        --deepspeed ds_config.json \
        --max_train_samples=10 \
        --max_eval_samples=10 \
        --optim adamw_ort_fused
```

## Register Model
### Register Onnx model

![image](https://github.com/savitamittal1/testing/assets/39776179/310782f8-2b9a-4f24-ada8-27bedd8a87da)

![image](https://github.com/savitamittal1/testing/assets/39776179/875238a5-f942-4047-8aef-6b77665bd9d7)

![image](https://github.com/savitamittal1/testing/assets/39776179/1535ab08-2884-4508-8d4a-06233ebe0f8f)

### Register MLFlow model

![image](https://github.com/savitamittal1/testing/assets/39776179/a41dee94-32a8-4d00-8223-d9957dc123c5)


## Model Evaluation

![image](https://github.com/savitamittal1/testing/assets/39776179/b80f1dc8-15b7-4e3f-8411-8250842ac5f1)

![image](https://github.com/savitamittal1/testing/assets/39776179/74502009-d99f-49b9-bf2f-3365e55f78f9)

![image](https://github.com/savitamittal1/testing/assets/39776179/d62e66dc-be28-4739-8680-639fdb7977fc)

![image](https://github.com/savitamittal1/testing/assets/39776179/f8428922-0153-4da2-8a92-3b554b11b142)




## Inference with Onnx Model

### Step 1: Deploy online endpoint

In your AML Workspace, go to the "Endpoints" tab. Create a new endpoint.

![image](https://github.com/savitamittal1/testing/assets/39776179/93c6525d-8e53-4448-b92a-cf56b26c71d9)

- Select the model you previously registered
- Select the environment you previously registered
- Add [score.py](score.py) when asked to provide scoring file
- Set your compute target (ACPT requires >=64GB of disk storage)
- Use all other defaults, deploy endpoint

### Invoke endpoint with test data 
```
{
    "inputs": {
        "article": ["summarize: The Inflation Reduction Act lowers prescription drug costs, health care costs, and energy costs. It's the most aggressive action on tackling the climate crisis in American history, which will lift up American workers and create good-paying, union jobs across the country. It'll lower the deficit and ask the ultra-wealthy and corporations to pay their fair share. And no one making under $400,000 per year will pay a penny more in taxes."],
    }
}
```


## FAQ
### Problem with Azure Authentication
If there's an Azure authentication issue, install Azure CLI [here](https://docs.microsoft.com/en-us/cli/azure/) and run `az login --use-device-code`
<br>Additionally, you can try replacing AzureCliCredential() in aml_submit.py with DefaultAzureCredential()
<br>You can learn more about Azure Identity authentication [here](https://learn.microsoft.com/en-us/python/api/azure-identity/azure.identity?view=azure-python)
