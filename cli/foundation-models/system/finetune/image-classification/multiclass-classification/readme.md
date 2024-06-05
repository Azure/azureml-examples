# Fine-tuning a model for Image Multi-class Classification task

You can launch a sample pipeline for image multi-class classification using `transformers_image_classification_pipeline` component.

For using this component, run the shell script file `bash ./hftransformers-fridgeobjects-multiclass-classification.sh`.

Currently following models are supported:
| Model Name | Source |
| ------ | ---------- |
| [microsoft-beit-base-patch16-224-pt22k-ft22k](https://ml.azure.com/registries/azureml/models/microsoft-beit-base-patch16-224-pt22k-ft22k/version/19) | azureml registry |
| [microsoft-swinv2-base-patch4-window12-192-22k](https://ml.azure.com/registries/azureml/models/microsoft-swinv2-base-patch4-window12-192-22k/version/20) | azureml registry |
| [facebook-deit-base-patch16-224](https://ml.azure.com/registries/azureml/models/facebook-deit-base-patch16-224/version/19) | azureml registry |
| [google-vit-base-patch16-224](https://ml.azure.com/registries/azureml/models/google-vit-base-patch16-224/version/17) | azureml registry |
| [Image classification models from Huggingface's Transformer library](https://huggingface.co/models?pipeline_tag=image-classification&library=transformers)| HuggingFace |
