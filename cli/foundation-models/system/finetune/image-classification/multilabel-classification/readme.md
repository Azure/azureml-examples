# Fine-tuning a model for Image Multi-label Classification task

You can launch a sample pipeline for image multi-label classification using `transformers_image_classification_pipeline` component.

For using this component, run the shell script file `bash ./hftransformers-fridgeobjects-multilabel-classification.sh`.

Currently following models are supported:
| Model Name | Source |
| ------ | ---------- |
| [microsoft-beit-base-patch16-224-pt22k-ft22k](https://ml.azure.com/registries/azureml-staging/models/microsoft-beit-base-patch16-224-pt22k-ft22k/version/3?tid=72f988bf-86f1-41af-91ab-2d7cd011db47#overview) | azureml-staging registry |
| [microsoft-swinv2-base-patch4-window12-192-22k](https://ml.azure.com/registries/azureml-staging/models/microsoft-swinv2-base-patch4-window12-192-22k/version/3?tid=72f988bf-86f1-41af-91ab-2d7cd011db47#overview) | azureml-staging registry |
| [facebook-deit-base-patch16-224](https://ml.azure.com/registries/azureml-staging/models/facebook-deit-base-patch16-224/version/3?tid=72f988bf-86f1-41af-91ab-2d7cd011db47#overview) | azureml-staging registry |
| [google-vit-base-patch16-224](https://ml.azure.com/registries/azureml-staging/models/google-vit-base-patch16-224/version/3?tid=72f988bf-86f1-41af-91ab-2d7cd011db47#overview) | azureml-staging registry |
| [Image classification models from Huggingface's Transformer library](https://huggingface.co/models?pipeline_tag=image-classification&library=transformers)| HuggingFace |
