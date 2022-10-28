# Azure Machine Learning Responsible AI Dashboard and Scorecard 

Read more about how to generate the Responsible AI (RAI) dashboard [here](https://learn.microsoft.com/en-us/azure/machine-learning/how-to-responsible-ai-dashboard-sdk-cli?tabs=yaml) and Responsible AI scorecard [here](https://learn.microsoft.com/en-us/azure/machine-learning/how-to-responsible-ai-scorecard).

The Responsible AI components are supported for MLflow models with `scikit-learn` flavor that are trained on `pandas.DataFrame`.
The components accept both models and SciKit-Learn pipelines as input as long as the model or pipeline implements `predict` and `predict_proba` functions that conforms to the `scikit-learn` convention.
If not compatible, you can wrap your model's prediction function into a wrapper class that transforms the output into the format that is supported (`predict` and `predict_proba` of `scikit-learn`), and pass that wrapper class to modules in this repo.

## Sample directory 📖

| Scenario | Dataset | Data type | RAI component included | Link to sample |
| --- | --- | --- | --- | --- |
| Regression | [sklearn Diabetes](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_diabetes.html) | Tabular | Explanation, Error Analysis, Counterfactuals | [responsibleaidashboard-diabetes-regression-model-debugging.ipynb](https://github.com/Azure/azureml-examples/blob/main/sdk/python/responsible-ai/responsibleaidashboard-diabetes-regression-model-debugging/responsibleaidashboard-diabetes-regression-model-debugging.ipynb) |
| Regression | [Programmers MLTable data](https://github.com/Azure/azureml-examples/tree/main/sdk/python/responsible-ai/responsibleaidashboard-programmer-regression-model-debugging/data-programmer-regression) | Tabular | Explanation, Error Analysis, Causal analysis, Counterfactuals | [responsibleaidashboard-programmer-regression-model-debugging.ipynb](https://github.com/Azure/azureml-examples/blob/main/sdk/python/responsible-ai/responsibleaidashboard-programmer-regression-model-debugging/responsibleaidashboard-programmer-regression-model-debugging.ipynb) |
| Classification | [Kaggle Housing](https://www.kaggle.com/alphaepsilon/housing-prices-dataset) | Tabular | Explanation, Error Analysis, Causal analysis, Counterfactuals | [responsibleaidashboard-housing-classification-model-debugging.ipynb](https://github.com/Azure/azureml-examples/blob/main/sdk/python/responsible-ai/responsibleaidashboard-housing-classification-model-debugging/responsibleaidashboard-housing-classification-model-debugging.ipynb) |
| Decision making | [Kaggle Housing](https://www.kaggle.com/alphaepsilon/housing-prices-dataset) | Tabular | Causal analysis, Counterfactuals | [responsibleaidashboard-housing-decision-making.ipynb](https://github.com/Azure/azureml-examples/blob/main/sdk/python/responsible-ai/responsibleaidashboard-housing-decision-making/responsibleaidashboard-housing-decision-making.ipynb) |
| Decision making | [sklearn Diabetes](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_diabetes.html) | Tabular | Causal analysis, Counterfactuals | [responsibleaidashboard-diabetes-decision-making.ipynb](https://github.com/Azure/azureml-examples/blob/main/sdk/python/responsible-ai/responsibleaidashboard-diabetes-decision-making/responsibleaidashboard-diabetes-decision-making.ipynb) |

## Supportability 🧰
Currently, we support datasets having numerical and categorical features. The following table provides the scenarios supported for each of the four responsible AI components:
> **Note**: Model overview (performance metrics and fairness disparity metrics) and Data explorer are generated for every Responsible AI dashboard by default and do not require a component to be configured.

| RAI component | Binary classification | Multi-class classification | Multilabel classification | Regression | Timeseries forecasting | Categorical features | Text features | Image Features | Recommender Systems | Reinforcement Learning |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | -- |
| Explainability | Yes | Yes | No | Yes | No | Yes | No | No | No | No |
| Error Analysis | Yes | Yes | No | Yes | No | Yes | No | No | No | No |
| Causal Analysis | Yes | No | No | Yes | No | Yes (max 5 features due to computational cost) | No | No | No | No |
| Counterfactual | Yes | Yes | No | Yes | No | Yes | No | No | No | No |

Read more about how to use the Responsible AI dashboard [here](https://learn.microsoft.com/en-us/azure/machine-learning/how-to-responsible-ai-dashboard).
