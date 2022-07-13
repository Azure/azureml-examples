import sklearn.datasets
from sklearn import linear_model
from sklearn.model_selection import train_test_split
import joblib
from pathlib import Path


alphas = [0.1, 0.5]
test_size = 0.3
model_dir = Path("models")
test_data_dir = Path("test-data")

diabetes = sklearn.datasets.load_diabetes(as_frame=True)
df = diabetes.frame

ycol = 'target'
Xcol = [c for c in df.columns if c != ycol]

X = df[Xcol]
y = df[ycol]

Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=test_size)

mods = {alpha : linear_model.Lasso(alpha=alpha) for alpha in alphas}

for alpha, mod in mods.items(): 
    mod.fit(Xtrain, ytrain)
    joblib.dump(mod, filename=model_dir / f"lasso-alpha-{alpha}.sav")

Xtest = Xtest.reset_index(drop=True)

with open(test_data_dir / "test-data-1.json", "w") as f:
    f.write(Xtest.loc[0:4,:].to_json())