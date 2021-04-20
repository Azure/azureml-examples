import mlflow
import mlflow.sklearn

from sklearn.datasets import load_diabetes
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

# Load and preprocess data
X, y = load_diabetes(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
data = {"train": {"X": X_train, "y": y_train}, "test": {"X": X_test, "y": y_test}}

# Enable auto logging
mlflow.autolog()

# Create, fit, and test the scikit-learn Ridge regression model
regression_model = Ridge(alpha=0.03)
regression_model.fit(data["train"]["X"], data["train"]["y"])
preds = regression_model.predict(data["test"]["X"])
print(preds)
