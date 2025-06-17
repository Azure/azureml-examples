# Azure Machine Learning Responsible AI Dashboard and Scorecard 

Read more about how to generate the Responsible AI (RAI) dashboard [here](https://learn.microsoft.com/en-us/azure/machine-learning/how-to-responsible-ai-dashboard-sdk-cli?tabs=yaml) and Responsible AI scorecard [here](https://learn.microsoft.com/en-us/azure/machine-learning/how-to-responsible-ai-scorecard).

The Responsible AI components are supported for MLflow models with `scikit-learn` flavor that are trained on `pandas.DataFrame`.
The components accept both models and SciKit-Learn pipelines as input as long as the model or pipeline implements `predict` and `predict_proba` functions that conforms to the `scikit-learn` convention.
If not compatible, you can wrap your model's prediction function into a wrapper class that transforms the output into the format that is supported (`predict` and `predict_proba` of `scikit-learn`), and pass that wrapper class to modules in this repo.

## Directory 📖
 



| Scenario | Dataset | Data type | RAI component included | Link to sample | Documentation |
| --- | --- | --- | --- | --- | --- |
| Regression | [sklearn Diabetes](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_diabetes.html) | Tabular | Explanation, Error Analysis, Counterfactuals | [responsibleaidashboard-diabetes-regression-model-debugging.ipynb](./tabular/responsibleaidashboard-diabetes-regression-model-debugging/responsibleaidashboard-diabetes-regression-model-debugging.ipynb) | [Tabular Dashboard Generation](https://learn.microsoft.com/en-us/azure/machine-learning/how-to-responsible-ai-dashboard?view=azureml-api-2) |
| Regression | [Programmers MLTable data](./tabular/responsibleaidashboard-programmer-regression-model-debugging/data-programmer-regression) | Tabular | Explanation, Error Analysis, Causal analysis, Counterfactuals | [responsibleaidashboard-programmer-regression-model-debugging.ipynb](./tabular/responsibleaidashboard-programmer-regression-model-debugging/responsibleaidashboard-programmer-regression-model-debugging.ipynb) | [Tabular Dashboard Generation](https://learn.microsoft.com/en-us/azure/machine-learning/how-to-responsible-ai-dashboard?view=azureml-api-2) |
| Binary Classification | [Finance Story](./tabular/responsibleaidashboard-finance-loan-classification/Fabricated_Loan_data.csv) | Tabular | Explanation, Error Analysis, Causal analysis, Counterfactuals | [Finance_Dashboard.ipynb](./tabular/responsibleaidashboard-finance-loan-classification/responsibleaidashboard-finance-loan-classification.ipynb) | [Documentation](./tabular/responsibleaidashboard-finance-loan-classification/readme.md) |
| Binary Classification | [Healthcare Story](./tabular/responsibleaidashboard-healthcare-covid-classification/data_covid_classification/) | Tabular | Explanation, Error Analysis, Causal analysis, Counterfactuals | [Covid_Healthcare_Dashboard.ipynb](./tabular/responsibleaidashboard-healthcare-covid-classification/responsibleaidashboard-healthcare-covid-classification.ipynb) | [Documentation](./tabular/responsibleaidashboard-healthcare-covid-classification/readme.md) |
| Binary Classification | [Education Story](./tabular/responsibleaidashboard-education-student-attrition-classificaton/Fabricated_Student_Attrition_Data.csv) | Tabular | Explanation, Error Analysis, Causal analysis, Counterfactuals | [Education_Dashboard.ipynb](./tabular/responsibleaidashboard-education-student-attrition-classificaton/responsibleaidashboard-education-student-attrition-classificaton.ipynb) | [Documentation](./tabular/responsibleaidashboard-education-student-attrition-classificaton/readme.md) |
| Classification | [Kaggle Housing](https://www.kaggle.com/alphaepsilon/housing-prices-dataset) | Tabular | Explanation, Error Analysis, Causal analysis, Counterfactuals | [responsibleaidashboard-housing-classification-model-debugging.ipynb](./tabular/responsibleaidashboard-housing-classification-model-debugging/responsibleaidashboard-housing-classification-model-debugging.ipynb) | [Tabular Dashboard Generation](https://learn.microsoft.com/en-us/azure/machine-learning/how-to-responsible-ai-dashboard?view=azureml-api-2) |
| Decision Making | [Kaggle Housing](https://www.kaggle.com/alphaepsilon/housing-prices-dataset) | Tabular | Causal analysis, Counterfactuals | [responsibleaidashboard-housing-decision-making.ipynb](./tabular/responsibleaidashboard-housing-decision-making/responsibleaidashboard-housing-decision-making.ipynb) | [Tabular Dashboard Generation](https://learn.microsoft.com/en-us/azure/machine-learning/how-to-responsible-ai-dashboard?view=azureml-api-2) |
| Decision Making | [sklearn Diabetes](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_diabetes.html) | Tabular | Causal analysis, Counterfactuals | [responsibleaidashboard-diabetes-decision-making.ipynb](./tabular/responsibleaidashboard-diabetes-decision-making/responsibleaidashboard-diabetes-decision-making.ipynb) | [Tabular Dashboard Generation](https://learn.microsoft.com/en-us/azure/machine-learning/how-to-responsible-ai-dashboard?view=azureml-api-2) |


To learn more about the different types of Dashboard visit the below tutorials:
1) [Tabular Dashboard Generation](https://learn.microsoft.com/en-us/azure/machine-learning/how-to-responsible-ai-dashboard?view=azureml-api-2)


## Supportability 🧰
Currently, we support datasets having numerical and categorical features. The following table provides the scenarios supported for each of the four responsible AI components:
> **Note**: Model overview (performance metrics and fairness disparity metrics) and Data explorer are generated for every Responsible AI dashboard by default and do not require a component to be configured.

| RAI component | Binary classification | Multi-class classification | Multilabel classification | Regression | Timeseries forecasting | Categorical features | Text features | Image Features | Recommender Systems | Reinforcement Learning |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | -- |
| Explainability | Yes | Yes | No | Yes | No | Yes | Yes | Yes | No | No |
| Error Analysis | Yes | Yes | No | Yes | No | Yes | Yes | Yes | No | No |
| Causal Analysis | Yes | No | No | Yes | No | Yes (max 5 features due to computational cost) | No | No | No | No |
| Counterfactual | Yes | Yes | No | Yes | No | Yes | No | No | No | No |

Read more about how to use the Responsible AI dashboards [here](https://learn.microsoft.com/en-us/azure/machine-learning/how-to-responsible-ai-dashboard). 

