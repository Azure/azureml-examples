# Healthcare AI Models

This directory contains deployment notebooks for Microsoft's Healthcare AI models that can be deployed and used with Azure Machine Learning or Azure AI Foundry. These notebooks provide you with the tools to quickly get started with healthcare-specific AI capabilities.


> ## Disclaimer
> **Important:** The Microsoft healthcare AI models, code and examples are intended for research and model development exploration. The models, code and examples are not designed or intended to be deployed in clinical settings as-is nor for use in the diagnosis or treatment of any health or medical condition, and the individual models' performances for such purposes have not been established. You bear sole responsibility and liability for any use of the healthcare AI models, code and examples, including verification of outputs and incorporation into any product or service intended for a medical purpose or to inform clinical decision-making, compliance with applicable healthcare laws and regulations, and obtaining any necessary clearances or approvals.

## Deployment Notebooks in This Directory

The following deployment notebooks are organized by model:

### MedImageInsight (MI2)
- [**MedImageInsight Deployment**](./medimageinsight_deploy.ipynb) - Deploy the MedImageInsight model for medical image understanding and analysis
- [**MedImageInsight Batch Deployment**](./medimageinsight_batch_deploy.ipynb) - Batch processing deployment for the MedImageInsight model

### MedImageParse (MIP)
- [**MedImageParse Deployment**](./medimageparse_deploy.ipynb) - Deploy the MedImageParse model for medical image segmentation
- [**MedImageParse Batch Deployment**](./medimageparse_batch_deploy.ipynb) - Batch processing deployment for the MedImageParse model

### CXRReportGen
- [**CXRReportGen Deployment**](./cxrreportgen_deploy.ipynb) - Deploy the CXRReportGen model for chest X-ray report generation
- [**CXRReportGen Batch Deployment**](./cxrreportgen_batch_deploy.ipynb) - Batch processing deployment for the CXRReportGen model

These notebooks will guide you through the process of deploying these healthcare AI models as endpoints that you can use in your applications or further experimentation.

## Healthcare AI Examples Repository

For more detailed examples, usage patterns, and solution templates demonstrating how to utilize these models, explore the [Healthcare AI Examples repository](https://aka.ms/healthcare-ai-examples). This repository provides practical examples and templates to help you get started with Microsoft's multimodal Healthcare AI models. The Healthcare AI Examples repository includes a helpful healthcareai_toolkit package that simplifies working with the deployed endpoints, DICOM files, and other common healthcare AI tasks.

This repository also contains numerous samples that guide you through scenarios like:

- Performing zero-shot and adapter-based classification of medical images using state-of-the-art embedding models
- Building image-search systems for fast and efficient retrieval of patient records based on pixel data from medical imagery
- Combining embedding models for multi-modal analysis

And more!
