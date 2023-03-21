# Using DeepSpeed MII for inference optimization

## What is Deepspeed MII?

DeepSpeed-MII is a new open-source python library from DeepSpeed, aimed towards making low-latency, low-cost inference of powerful models not only feasible but also easily accessible.

* MII offers access to highly optimized implementation of thousands of widely used DL models.
* MII supported models achieve significantly lower latency and cost compared to their original implementation. For example, MII reduces the latency of Big-Science Bloom 176B model by 5.7x, while reducing the cost by over 40x. Similarly, it reduces the latency and cost of deploying Stable Diffusion by 1.9x. See more details for [an exhaustive latency and cost analysis of MII](#quantifying-latency-and-cost-reduction).
* To enable low latency/cost inference, MII leverages an extensive set of optimizations from DeepSpeed-Inference such as deepfusion for transformers, automated tensor-slicing for multi-GPU inference, on-the-fly quantization with ZeroQuant, and several others (see our [blog post](https://www.deepspeed.ai/2022/10/10/mii.html) for more details).
* With state-of-the-art performance, MII supports low-cost deployment of these models both on-premises and on Azure via AzureML with just a few lines of codes.

## How does MII work?

![Text Generation Models](https://github.com/microsoft/DeepSpeed-MII/blob/main/docs/images/mii-arch.png?raw=true)


*Figure 1: MII Architecture, showing how MII automatically optimizes OSS models using DS-Inference before deploying them on-premises using GRPC, or on Microsoft Azure using AzureML Inference.*

Under-the-hood MII is powered by [DeepSpeed-Inference](https://arxiv.org/abs/2207.00032). Based on model type, model size, batch size, and available hardware resources, MII automatically applies the appropriate set of system optimizations from DeepSpeed-Inference to minimize latency and maximize throughput. It does so by using one of many pre-specified model injection policies, that allows MII and DeepSpeed-Inference to identify the underlying PyTorch model architecture and replace it with an optimized implementation (see *Figure A*). In doing so, MII makes the expansive set of optimizations in DeepSpeed-Inference automatically available for thousands of popular models that it supports.

## Supported Models and Tasks

MII currently supports over 30,000 models across a range of tasks such as text-generation, question-answering, text-classification. The models accelerated by MII are available through multiple open-sourced model repositories such as Hugging Face, FairSeq, EluetherAI, etc. We support dense models based on Bert, Roberta or GPT **architectures** ranging from few hundred million parameters to tens of billions of parameters in size. We continue to expand the list with support for massive hundred billion plus parameter dense and sparse models coming soon.

MII model support will continue to grow over time, check back for updates! Currently we support the following Hugging Face Transformers model families:

model family | size range | ~model count
------ | ------ | ------
[bloom](https://huggingface.co/models?other=bloom) | 0.3B - 176B | 198
[stable-diffusion](https://huggingface.co/models?other=stable-diffusion) | 1.1B | 330
[opt](https://huggingface.co/models?other=opt) | 0.1B - 66B | 170
[gpt\_neox](https://huggingface.co/models?other=gpt_neox) | 1.3B - 20B | 37
[gptj](https://huggingface.co/models?other=gptj) | 1.4B - 6B | 140
[gpt\_neo](https://huggingface.co/models?other=gpt_neo) | 0.1B - 2.7B | 300
[gpt2](https://huggingface.co/models?other=gpt2) | 0.3B - 1.5B | 7,888
[xlm-roberta](https://huggingface.co/models?other=xlm-roberta) | 0.1B - 0.3B | 1,850
[roberta](https://huggingface.co/models?other=roberta) | 0.1B - 0.3B | 5,190
[bert](https://huggingface.co/models?other=bert) | 0.1B - 0.3B | 13,940


# MII-Public and MII-Azure

MII can work with two variations of DeepSpeed-Inference. The first, referred to as ds-public, contains most of the DeepSpeed-Inference optimizations discussed here,  is also available via our open-source DeepSpeed library. The second referred to as ds-azure, offers tighter integration with Azure, and is available via MII to all Microsoft Azure customers. We refer to MII running the two DeepSpeed-Inference variants as MII-Public and MII-Azure, respectively.

While both variants offers significant latency and cost reduction over the open-sourced PyTorch baseline, the latter, offers additional performance advantage for generation based workloads. The full latency and cost advantage comparison with PyTorch baseline and across these two versions is available [here](#quantifying-latency-and-cost-reduction).

# Getting Started with MII

## Installation

We regularly push releases to [PyPI](https://pypi.org/project/deepspeed-mii/) and encourage users to install from there in most cases.

```bash
pip install deepspeed-mii
```

## Deploying MII-Public

MII-Public can be deployed on-premises or on any cloud offering with just a few lines of code. MII creates a lightweight GRPC server to support this form of deployment and provides a GRPC inference endpoint for queries.

Several deployment and query examples can be found here: [examples/local](https://github.com/microsoft/DeepSpeed-MII/tree/main/examples/local)

As an example here is a deployment of the [bigscience/bloom-560m](https://huggingface.co/bigscience/bloom-560m) model from Hugging Face:

**Deployment**
```python
import mii
mii_configs = {"tensor_parallel": 1, "dtype": "fp16"}
mii.deploy(task="text-generation",
           model="bigscience/bloom-560m",
           deployment_name="bloom560m_deployment",
           mii_config=mii_configs)
```

This will deploy the model onto a single GPU and start the GRPC server that can later be queried.

**Query**
```python
import mii
generator = mii.mii_query_handle("bloom560m_deployment")
result = generator.query({"query": ["DeepSpeed is", "Seattle is"]}, do_sample=True, max_new_tokens=30)
print(result)
```

The only required key is `"query"`, all other items outside the dictionary will be passed to `generate` as kwargs. For Hugging Face provided models you can find all possible arguments in their [documentation for generate](https://huggingface.co/docs/transformers/v4.20.1/en/main_classes/text_generation#transformers.generation_utils.GenerationMixin.generate).

**Shutdown Deployment**
```python
import mii
mii.terminate("bloom560m_deployment")
```

## Deploying with MII-Azure

MII supports deployment on Azure via AzureML Inference. To enable this, MII generates AzureML deployment assets for a given model that can be deployed using the Azure-CLI, as shown in the code below. Furthermore, deploying on Azure, allows MII to leverage DeepSpeed-Azure as its optimization backend, which offers better latency and cost reduction than DeepSpeed-Public.

This deployment process is very similar to local deployments and we will modify the code from the local deployment example with the [bigscience/bloom-560m](https://huggingface.co/bigscience/bloom-560m) model.

---
📌 **Note:**  MII-Azure has the benefit of supporting DeepSpeed-Azure for better latency and cost than DeepSpeed-Public for certain workloads. We are working to enable DeepSpeed-Azure automatically for all MII-Azure deployments in a near-term MII update. In the meantime, we are offering DeepSpeed-Azure as a preview release to MII-Azure users. If you have a MII-Azure deployment and would like to try DeepSpeed-Azure, please reach out to us at deepspeed-mii@microsoft.com to get access.

---

Several other AzureML deployment examples can be found here: [examples/aml](https://github.com/microsoft/DeepSpeed-MII/tree/main/examples/aml)

**Setup**

To use MII on AzureML resources, you must have the Azure-CLI installed with an active login associated with your Azure resources. Follow the instructions below to get your local system ready for deploying on AzureML resources:

1. Install Azure-CLI. Follow the official [installation instructions](https://learn.microsoft.com/en-us/cli/azure/install-azure-cli#install).
2. Run `az login` and follow the instructions to login to your Azure account. This account should be linked to the resources you plan to deploy on.
3. Set the default subscription with `az account set --subscription <YOUR-SUBSCRIPTION-ID>`. You can find your subscription ID in the "overview" tab on your resource group page from the Azure web portal.
4. Install the AzureML plugin for Azure-CLI with `az extension add --name ml`

**Deployment**
```python
import mii
mii_configs = {"tensor_parallel": 1, "dtype": "fp16"}
mii.deploy(task="text-generation",
           model="bigscience/bloom-560m",
           deployment_name="bloom560m-deployment",
           deployment_type=mii.constants.DeploymentType.AML,
           mii_config=mii_configs)
```

---
📌 **Note:** Running the `mii.deploy` with `deployment_type=mii.constants.DeploymentType.AML` will only generate the scripts to launch an AzureML deployment. You must also run the generated `deploy.sh` script to run on AzureML resources.

---

This will generate the scripts and configuration files necessary to deploy the model on AzureML using a single GPU. You can find the generated output at `./bloom560m-deployment_aml/`

When you are ready to run your deployment on AzureML resources, navigate to the newly created directory and run the deployment script:
```bash
cd ./bloom560m-deployment_aml/
bash deploy.sh
```

This script may take several minutes to run as it does the following:
- Downloads the model locally
- Creates a Docker Image with MII for your deployment
- Creates an AzureML online-endpoint for running queries
- Uploads and registers the model to AzureML
- Starts your deployment

---
📌 **Note:** Large models (e.g., `bigscience/bloom`) may cause a timeout when trying to upload and register the model to AzureML. In these cases, it is required to manually upload models to Azure blob storage with [AzCopy](https://learn.microsoft.com/en-us/azure/storage/common/storage-use-azcopy-v10). Instructions and automation of this step will be added soon.

---

**Query**
Once the deployment is running on AzureML, you can run queries by navigating to the online-endpoint that was created for this deployment (i.e., `bloom-560m-deployment-endpoint`) from the [AzureML web portal](https://ml.azure.com/endpoints). Select the "Test" tab at the top of the endpoint page and type your query into the text-box:
```
{"query": ["DeepSpeed is", "Seattle is"], "do_sample"=True, "max_new_tokens"=30}
```

The only required key is `"query"`, all other items in the dictionary will be passed to `generate` as kwargs. For Hugging Face provided models you can find all possible arguments in their [documentation for generate](https://huggingface.co/docs/transformers/v4.20.1/en/main_classes/text_generation#transformers.generation_utils.GenerationMixin.generate).


# Quantifying Latency and Cost Reduction

Inference workloads can be either latency critical, where the primary objective is to minimize latency, or cost sensitive, where the primary objective is to minimize cost. In this section, we quantify the benefits of using MII for both latency-critical and cost-sensitive scenarios.

## Latency Critical Scenarios

For latency-critical scenarios, where a small batch size of 1 is often used, MII can reduce the latency by up to 6x for a wide range of open-source models, across multiple tasks. More specifically, we show model latency reduction of [^overhead_details]:

1. Up to 5.7x for multi-GPU inference for text generation using massive models such as Big Science Bloom, Facebook OPT, and EluetherAI NeoX (*Figure 2 (left)*)

2. Up to 1.9x for image generation tasks model using Stable Diffusion (*Figure 2 (right)*)

3. Up to 3x for relatively smaller text generation models (up to 7B parameters) based on OPT, BLOOM, and GPT architectures, running on a single GPU (*Figures 3 and 4*)

4. Up to 9x for various text representation tasks like fill-mask, text classification, question answering, and token classification using RoBERTa- and BERT- based models (*Figures 5 and 6*).

[ ![multi gpu latency](https://github.com/microsoft/DeepSpeed-MII/blob/main/docs/images/llm-latency-sd-latency.png?raw=True) ](https://github.com/microsoft/DeepSpeed-MII/blob/main/docs/images/llm-latency-sd-latency-zoom.png?raw=True)
*Figure 2: (Left) Best achievable latency for large models. MII-Azure (int8) offers 5.7X lower latency compared to Baseline for Bloom-176B. (Right) Stable Diffusion text to image generation latency comparison.*

[ ![OPT and BLOOM Models](https://github.com/microsoft/DeepSpeed-MII/blob/main/docs/images/opt-bloom.png?raw=True) ](https://github.com/microsoft/DeepSpeed-MII/blob/main/docs/images/opt-bloom.png?raw=True)
*Figure 3: Latency comparison for OPT and BLOOM models. MII-Azure is up to 2.8x faster than baseline.*

[ ![GPT Models](https://github.com/microsoft/DeepSpeed-MII/blob/main/docs/images/gpt.png?raw=True) ](https://github.com/microsoft/DeepSpeed-MII/blob/main/docs/images/mii/gpt.png?raw=True)
*Figure 4: Latency comparison for GPT models. MII-Azure is up to 3x faster than baseline.*

[ ![Roberta Models](https://github.com/microsoft/DeepSpeed-MII/blob/main/docs/images/roberta.png?raw=True) ](https://github.com/microsoft/DeepSpeed-MII/blob/main/docs/images/roberta.png?raw=True)
*Figure 5: Latency comparison for RoBERTa models. MII offers up to 9x lower model latency and up to 3x lower end-to-end latency than baseline on several tasks and RoBERTa variants [^overhead_details].*

[ ![Bert Models](https://github.com/microsoft/DeepSpeed-MII/blob/main/docs/images/bert.png?raw=True) ](https://github.com/microsoft/DeepSpeed-MII/blob/main/docs/images/bert.png?raw=True)
*Figure 6: Latency comparison for BERT models. MII offers up to 8.9x lower model latency and up to 4.5x end-to-end latency across several tasks and BERT variants[^overhead_details].*

[^overhead_details]: The end-to-end latency of an inference workload is comprised of two components: i) actual model execution, and ii) pre-/post-processing before and after the model execution. MII optimizes the actual model execution but leaves the pre-/post-processing pipeline for future optimizations. We notice that text representation tasks have significant pre-/post-processing overhead (*Figures G and H*). We plan to address those in a future update.

## Cost Sensitive Scenarios

MII can significantly reduce the inference cost of very expensive language models like Bloom, OPT, etc. To get the lowest cost, we use a large batch size that maximizes throughput for both baseline and MII. Here we look at the cost reduction from MII using two different metrics: i) tokens generated per second per GPU, and ii) dollars per million tokens generated.

*Figures 7 and 8* show that MII-Public offers over 10x throughput improvement and cost reduction compared to the baseline, respectively. Furthermore, MII-Azure offers over 30x improvement in throughput and cost compared to the baseline.

[ ![tput large models](https://github.com/microsoft/DeepSpeed-MII/blob/main/docs/images/tput-llms.png?raw=True) ](https://github.com/microsoft/DeepSpeed-MII/blob/main/docs/images/tput-llms.png?raw=True)
*Figure 7: Throughput comparison per A100-80GB GPU for large models. MII-Public offers over 15x throughput improvement while MII-Azure offers over 40x throughput improvement.*

[ ![azure cost](https://github.com/microsoft/DeepSpeed-MII/blob/main/docs/images/azure-cost.png?raw=True) ](https://github.com/microsoft/DeepSpeed-MII/blob/main/docs/images/azure-cost.png?raw=True)
*Figure 8: Cost of generating 1 million tokens on Azure with different model types. MII-Azure reduces the cost of generation by over 40x.*

# Community **Tutorials**

* [DeepSpeed Deep Dive — Model Implementations for Inference (MII) (Heiko Hotz)](https://towardsdatascience.com/deepspeed-deep-dive-model-implementations-for-inference-mii-b02aa5d5e7f7)

