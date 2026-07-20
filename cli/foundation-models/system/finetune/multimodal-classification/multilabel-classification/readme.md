# Fine-tuning a model for Multimodal Multi-label Classification task

You can launch a sample pipeline for multimodal multi-label classification using `multimodal_classification_pipeline` component.

For using this component, run the shell script file `bash ./mmeft-chxray-multilabel-classification.sh`.

Currently following models are supported:<br />
| Model Name | Source |<br />
| ---------- | ---------- |<br />
| [mmeft](https://ml.azure.com/registries/azureml/models/mmeft/version/1) | azureml registry |

### Training data used:
We will use the [ChXray](https://automlresources-prod.azureedge.net/datasets/ChXray.zip) dataset.
Original source of dataset: https://nihcc.app.box.com/v/ChestXray-NIHCC/file/220660789610

[arXiv:1705.02315](https://arxiv.org/abs/1705.02315v5) [cs.CV]