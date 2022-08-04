import argparse
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report
import os
import pandas as pd
import mlflow


def select_first_file(path):
    """Selects first file in folder, use under assumption there is only one file in folder
    Args:
        path (str): path to directory or file to choose
    Returns:
        str: full path of selected file
    """
    files = os.listdir(path)
    return os.path.join(path, files[0])


# Start Logging
mlflow.start_run()

# enable autologging
# mlflow.sklearn.autolog()

os.makedirs("./outputs", exist_ok=True)


def main():
    """Main function of the script."""

    # input and output arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_data", type=str, help="path to train data")
    parser.add_argument("--test_data", type=str, help="path to test data")
    parser.add_argument("--n_estimators", required=False, default=100, type=int)
    parser.add_argument("--learning_rate", required=False, default=0.1, type=float)
    parser.add_argument("--model", type=str, help="path to model file")
    args = parser.parse_args()

    # paths are mounted as folder, therefore, we are selecting the file from folder
    train_df = pd.read_csv(select_first_file(args.train_data))

    # Extracting the label column
    y_train = train_df.pop("default payment next month")

    # convert the dataframe values to array
    X_train = train_df.values

    # paths are mounted as folder, therefore, we are selecting the file from folder
    test_df = pd.read_csv(select_first_file(args.test_data))

    # Extracting the label column
    y_test = test_df.pop("default payment next month")

    # convert the dataframe values to array
    X_test = test_df.values

    print(f"Training with data of shape {X_train.shape}")

    clf = GradientBoostingClassifier(
        n_estimators=args.n_estimators, learning_rate=args.learning_rate
    )
    clf.fit(X_train, y_train)

    test_accuracy = clf.score(X_test, y_test)

    mlflow.log_metric("test_accuracy_score", test_accuracy)

    # # Registering the model to the workspace
    # print("Registering the model via MLFlow")
    # mlflow.sklearn.log_model(
    #     sk_model=clf,
    #     registered_model_name=args.registered_model_name,
    #     artifact_path=args.registered_model_name,
    # )

    # Saving the model to a file
    mlflow.sklearn.save_model(
        sk_model=clf,
        path=os.path.join(args.model, "trained_model"),
    )

    # Stop Logging
    mlflow.end_run()


if __name__ == "__main__":
    main()
