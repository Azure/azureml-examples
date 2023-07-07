# Fine-tuning a model for Image Multi-class Classification task

You can launch a sample pipeline for image multi-class classification using `transformers_image_classification_pipeline` component.

For using this component, run the shell script file `bash ./hftransformers-fridgeobjects-multiclass-classification.sh`.

Currently following models are supported:
| Model Name | Source |
| ------ | ---------- |
| [microsoft-beit-base-patch16-224-pt22k-ft22k](https://ml.azure.com/registries/azureml-preview/models/microsoft-beit-base-patch16-224-pt22k-ft22k/version/1?tid=72f988bf-86f1-41af-91ab-2d7cd011db47#overview) | azureml-preview registry |
| [microsoft-swinv2-base-patch4-window12-192-22k](https://ml.azure.com/registries/azureml-preview/models/microsoft-swinv2-base-patch4-window12-192-22k/version/1?tid=72f988bf-86f1-41af-91ab-2d7cd011db47#overview) | azureml-preview registry |
| [facebook-deit-base-patch16-224](https://ml.azure.com/registries/azureml-preview/models/facebook-deit-base-patch16-224/version/1?tid=72f988bf-86f1-41af-91ab-2d7cd011db47#overview) | azureml-preview registry |
| [google-vit-base-patch16-224](https://ml.azure.com/registries/azureml-preview/models/google-vit-base-patch16-224/version/1?tid=72f988bf-86f1-41af-91ab-2d7cd011db47#overview) | azureml-preview registry |
| [Image classification models from Huggingface](https://huggingface.co/models?pipeline_tag=image-classification&sort=downloads) | HuggingFace |