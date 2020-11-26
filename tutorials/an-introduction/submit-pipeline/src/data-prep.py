import argparse
import os
import pandas as pd
import pickle
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

# define functions
def preprocess_data(df):
    X = df.drop(["species"], axis=1)
    y = df["species"]

    enc = LabelEncoder()
    y = enc.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    return X_train, X_test, y_train, y_test, enc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-data")
    parser.add_argument("--output-path")
    args = parser.parse_args()

    print("loading data...")
    df = pd.read_csv(args.input_data)
    print(df.head())

    # preprocess data
    print("preprocessing data...")
    X_train, X_test, y_train, y_test, enc = preprocess_data(df)

    if not (args.output_path is None):
        os.makedirs(args.output_path, exist_ok=True)
        print("%s created" % args.output_path)
    
    print("writing processed data to pickle file")
    processed_data = os.path.join(args.output_path, "processed_data.pkl")
    with open(processed_data, 'wb') as f:
        pickle.dump([X_train, X_test, y_train, y_test, enc], f)

if __name__ == "__main__":
    main()
