import mlflow
from mlflow.entities import Metric
from mlflow.tracking import MlflowClient
import numpy as np
import time
from sklearn import datasets
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
    plot_confusion_matrix,
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split


def normalized_confusion_matrix(model, X_test, y_test, display_labels, file_name):
    """
    Generated normalized confusion matrix using scikit-learn and saves as a figure
    """
    cm_normalized_plot = plot_confusion_matrix(
        model, X_test, y_test, display_labels=display_labels, normalize="true"
    )
    cm_normalized_plot.ax_.set_title("Normalized Confusion Matrix")
    cm_normalized_plot.figure_.savefig(file_name)


def main():
    # getting iris dataset to play with
    data = datasets.load_iris()

    # splitting dataset between training and test data
    iris_X, iris_y = data.data, data.target
    X_train, X_test, y_train, y_test = train_test_split(iris_X, iris_y)

    n_neighbors = 10

    lr = KNeighborsClassifier(n_neighbors=n_neighbors)
    lr.fit(X_train, y_train)

    y_predict = lr.predict(X_test)

    np.savetxt("predictions.csv", y_predict, delimiter=",")

    # logging table
    # saving data in a csv file and logging it as an artifact
    mlflow.log_artifact("predictions.csv", f"tables/predictions.csv")

    normalized_cm_file = "normalized_confusion_matrix.png"
    normalized_confusion_matrix(
        lr, X_test, y_test, data.target_names, normalized_cm_file
    )

    # logging confusion matrix
    # generating a plot of confusion matrix and logging it
    mlflow.log_artifact(normalized_cm_file, f"plots/{normalized_cm_file}")

    # logging param
    mlflow.log_param("n_neighbors", n_neighbors)

    # logging metric
    mlflow.log_metric("dummy_metric", 5.9)

    mlflow.sklearn.log_model(
        lr,
        artifact_path="model",
        registered_model_name="sk-learn-knn-model",
    )


# run script
if __name__ == "__main__":
    # add space in logs
    print("*" * 60)
    print("\n\n")

    # run main function
    main()

    # add space in logs
    print("*" * 60)
    print("\n\n")
