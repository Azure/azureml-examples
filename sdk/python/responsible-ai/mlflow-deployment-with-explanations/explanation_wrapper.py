import mlflow
import sys
import subprocess
import pathlib
import pandas as pd

from ml_wrappers.model.wrapped_classification_without_proba_model import (
    WrappedClassificationWithoutProbaModel,
)
from ml_wrappers.model.wrapped_classification_model import WrappedClassificationModel
from ml_wrappers.model.wrapped_regression_model import WrappedRegressionModel
from ml_wrappers.dataset import DatasetWrapper
from responsibleai import RAIInsights, FeatureMetadata
from raiutils.exceptions import UserConfigValidationException

REGRESSION_TASK_TYPE = "regression"
CLASSIFICATION_TASK_TYPE = "classification"


class ExplanationWrapper(mlflow.pyfunc.PythonModel):
    """Class to provide local explanations with predictions"""

    def __init__(
        self,
        model_uri,
        baseline_df,
        target_column,
        task_type,
        categorical_features=None,
    ):
        super().__init__()

        # Determine categorical features
        categorical_features = categorical_features
        if not categorical_features:
            categorical_features = compute_categorical_features(
                baseline_df, target_column
            )

        # Determine classes
        classes = None
        if task_type == CLASSIFICATION_TASK_TYPE:
            classes = list(baseline_df[target_column].unique())

        # Load model
        model = mlflow.pyfunc.load_model(model_uri)

        # Wrap model
        wrapped_model = wrap_model(
            model, task_type, baseline_df, target_column, classes
        )

        # Create RAI Insights
        feature_metadata = FeatureMetadata(
            categorical_features=categorical_features, dropped_features=[]
        )
        rai_insights = RAIInsights(
            model=wrapped_model,
            train=baseline_df,
            test=baseline_df,
            target_column=target_column,
            task_type=task_type,
            classes=classes,
            feature_metadata=feature_metadata,
        )
        rai_insights.explainer.add()

        # Save RAI Insights
        rai_insights.save("artifacts/RAI_Insights")

    def load_context(self, context):
        """This method is called when loading an MLFlow model with pyfunc.load_model(), as soon as the Python Model is constructed
        Args:
            context: MLFlow context where the model artifact is stored
        """
        # Load model dependencies
        formatted_path = pathlib.PureWindowsPath(context.artifacts["model"]).as_posix()
        try:
            conda_file = mlflow.pyfunc.get_model_dependencies(
                formatted_path, format="conda"
            )
            # call conda env update in subprocess
            subprocess.check_call(
                [sys.executable, "-m", "conda", "env", "update", "-f", conda_file]
            )
        except Exception as e:
            print("Failed to install model dependencies")
            print(e)

        # Load model
        self.model = mlflow.pyfunc.load_model(formatted_path)

        # Load explainer
        self.rai_insights = RAIInsights.load(
            pathlib.PureWindowsPath(context.artifacts["RAI insights"]).as_posix()
        )

    def predict(self, context, model_input):
        """This is an abstract function. It is customized to provide explanations

        :param context: the MLflow context where the model artifact is stored
        :type Map
        :param model_input: the input data to run through the model
        :type array-like
        :return: Map with keys 'predictions' and 'explanations'
        :type Map
        """
        # Validate data format
        if not isinstance(model_input, pd.DataFrame):
            raise UserConfigValidationException(
                "Data is of type {0} but it must be "
                "a pandas DataFrame.".format(type(data))
            )

        # Compute explanations
        explanations = []
        for index, data in model_input.iterrows():
            localExplanation = self.rai_insights.explainer.request_explanations(
                local=True, data=pd.DataFrame(data).T
            )
            explanations.append(
                localExplanation.precomputedExplanations.localFeatureImportance.scores
            )

        # Return payload
        return {
            "predictions": self.model.predict(model_input).tolist(),
            "explanations": explanations,
        }


def wrap_model(model, task_type, baseline_df, target_column, classes=None):
    if task_type == CLASSIFICATION_TASK_TYPE:
        if not hasattr(model, "predict_proba"):
            if not hasattr(model, "classes_"):
                model.classes_ = classes
            return WrappedClassificationWithoutProbaModel(model)
        else:
            return WrappedClassificationModel(
                model,
                model.predict_proba,
                DatasetWrapper(baseline_df.drop(columns=[target_column])),
            )
    else:
        return WrappedRegressionModel(
            model,
            model.predict,
            DatasetWrapper(baseline_df.drop(columns=[target_column])),
        )


def compute_categorical_features(baseline_dataframe, target_column):
    """Compute which features are categorical based on data type of the columns.

    :param baseline_dataframe: The baseline data meaning the data used to create the
    model monitor
    :type baseline_dataframe: pandas.DataFrame
    :param target_column: the column to predict
    :type target_column: string
    :return: categorical features
    :rtype: list[string]
    """
    categorical_features = []
    for column in baseline_dataframe.columns:
        baseline_column = pd.Series(baseline_dataframe[column])
        if (
            baseline_column.dtype.name == "object"
            and baseline_column.name != target_column
        ):
            categorical_features.append(baseline_column.name)
    return categorical_features
