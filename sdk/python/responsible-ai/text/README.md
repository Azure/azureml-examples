---
page_type: sample
languages:
- python
products:
- azure-machine-learning
description: This sample shows how to create responsible ai dashboard for text datasets.
---

## Azure Machine Learning Responsible AI Dashboard and Scorecard 

### overview

Read more about how to generate the Responsible AI (RAI) dashboard [here](https://learn.microsoft.com/en-us/azure/machine-learning/how-to-responsible-ai-dashboard-sdk-cli?tabs=yaml) and Responsible AI scorecard [here](https://learn.microsoft.com/en-us/azure/machine-learning/how-to-responsible-ai-scorecard).

The Responsible AI components are supported for MLflow models with `scikit-learn` flavor that are trained on `pandas.DataFrame`.
The components accept both models and SciKit-Learn pipelines as input as long as the model or pipeline implements `predict` and `predict_proba` functions that conforms to the `scikit-learn` convention.
If not compatible, you can wrap your model's prediction function into a wrapper class that transforms the output into the format that is supported (`predict` and `predict_proba` of `scikit-learn`), and pass that wrapper class to modules in this repo.

### objective
The main objective of this tutorial is to help users understand the process of creating responsible ai dashboard with explanations & error analysis for text dataset.

### programming languages
 - Python

### directory 📖

| Scenario | Dataset | Data type | RAI component included | Link to sample | Documentation |
| --- | --- | --- | --- | --- | --- |
| Text Classification | [DBPedia dataset](https://huggingface.co/datasets/DeveloperOats/DBPedia_Classes) | Text | Explanation, Error Analysis | [responsibleaidashboard-text-classification-DBPedia.ipynb](./text/responsibleaidashboard-text-classification-DBPedia.ipynb) | [Text Dashboard Generation](https://learn.microsoft.com/en-us/azure/machine-learning/how-to-responsible-ai-text-dashboard?view=azureml-api-2) |
| Multi-label Text Classification | [Covid19 Emergency Event Dataset](https://huggingface.co/datasets/joelito/covid19_emergency_event) | Text | Explanation, Error Analysis | [responsibleaidashboard-multilabel-text-classification-covid-events.ipynb](./text/responsibleaidashboard-multilabel-text-classification-covid-events.ipynb) | [Text Dashboard Generation](https://learn.microsoft.com/en-us/azure/machine-learning/how-to-responsible-ai-text-dashboard?view=azureml-api-2) |
| Binary Text Classification | [blbooksgenre dataset](https://huggingface.co/datasets/blbooksgenre) | Text | Explanation, Error Analysis | [responsibleaidashboard-text-classification-blbooksgenre.ipynb](./responsibleaidashboard-text-classification-blbooksgenre.ipynb) | [Text Dashboard Generation](https://learn.microsoft.com/en-us/azure/machine-learning/how-to-responsible-ai-text-dashboard?view=azureml-api-2) |
| Text Classification | [News Article Classification](./text/responsibleaidashboard-text-classification-financial-news/Text_classification_dataset.xlsx) | Text | Explanation, Error Analysis | [Financial_News_Text_classifier.ipynb](./text/responsibleaidashboard-text-classification-financial-news/responsibleaidashboard-text-classification-financial-news.ipynb) | [Documentation](./text/responsibleaidashboard-text-classification-financial-news/readme.md) |
| Text Question Answering | [Stanford Question Answering Dataset (SQuAD)](https://huggingface.co/datasets/squad) | Text | Explanation, Error Analysis | [responsibleaidashboard-text-question-answering-squad.ipynb](./text/responsibleaidashboard-text-question-answering-squad.ipynb) | [Text Dashboard Generation](https://learn.microsoft.com/en-us/azure/machine-learning/how-to-responsible-ai-text-dashboard?view=azureml-api-2)|


To learn more about the different types of Dashboard visit the tutorial -
[Text Dashboard Generation](https://learn.microsoft.com/en-us/azure/machine-learning/how-to-responsible-ai-text-dashboard?view=azureml-api-2)

### estimated runtime

### supportability 🧰
Currently, we support datasets having numerical and categorical features. The following table provides the scenarios supported for each of the four responsible AI components:
> **Note**: Model overview (performance metrics and fairness disparity metrics) and Data explorer are generated for every Responsible AI dashboard by default and do not require a component to be configured.

| RAI component | Binary classification | Multi-class classification | Multilabel classification | Regression | Timeseries forecasting | Categorical features | Text features | Image Features | Recommender Systems | Reinforcement Learning |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | -- |
| Explainability | Yes | Yes | No | Yes | No | Yes | Yes | Yes | No | No |
| Error Analysis | Yes | Yes | No | Yes | No | Yes | Yes | Yes | No | No |
| Causal Analysis | Yes | No | No | Yes | No | Yes (max 5 features due to computational cost) | No | No | No | No |
| Counterfactual | Yes | Yes | No | Yes | No | Yes | No | No | No | No |

Read more about how to use the Responsible AI dashboards [here](https://learn.microsoft.com/en-us/azure/machine-learning/how-to-responsible-ai-dashboard). 

