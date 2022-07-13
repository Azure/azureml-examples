import sklearn.datasets
from sklearn import linear_model
from sklearn.model_selection import train_test_split
import joblib
from pathlib import Path
import json 

test_size = 0.3
model_dir = Path("models")
test_data_dir = Path("test-data")

datasets = {'diabetes' : sklearn.datasets.load_diabetes(as_frame=True).frame,
            'iris' : sklearn.datasets.load_iris(as_frame=True).frame}

def train_model(name, dataset, test_size=0.3):
    ycol = 'target'
    Xcol = [c for c in dataset.columns if c != ycol]
    X = dataset[Xcol]
    y = dataset[ycol]
    Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=test_size)
    Xtest = Xtest.reset_index(drop=True)
    model = linear_model.Lasso(alpha=0.1) 
    model.fit(Xtrain, ytrain)
    joblib.dump(model, filename=model_dir / f"{name}.sav")
    with open(test_data_dir / f"{name}-test-data.json", "w") as f:
        test_data = {}
        test_data['model'] = name
        test_data['data'] = Xtest.loc[0:4,:].to_dict()
        f.write(json.dumps(test_data))

for name, dataset in datasets.items():
    train_model(name, dataset)