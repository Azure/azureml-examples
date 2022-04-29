from dask.distributed import Client, LocalCluster
import dask.dataframe as dd
import os, uuid, sys
import argparse
import mlflow
import tempfile
import shutil
import xgboost as xgb

def write_freeze():
    # log pip list before doing anything else
    from pathlib import Path
    import os

    Path("./outputs").mkdir(parents=True, exist_ok=True)
    os.system("pip list > outputs/pip_list.txt")

if __name__ == '__main__':
    write_freeze()
    
    print("Command Line Parameters: ", sys.argv)

    for k, v in os.environ.items():
        if k.startswith("MLFLOW"):
            print(k, v)

    parser = argparse.ArgumentParser()
    parser.add_argument("--nyc_taxi_parquet")   # input folder
    parser.add_argument("--model")              # output folder
    parser.add_argument("--tree_method")        # auto, exact, approx, hist
    parser.add_argument("--learning_rate", type=float)      # 0.3; range: [0,1]
    parser.add_argument("--gamma", type=float)              # 0; range: [0,inf]
    parser.add_argument("--max_depth", type=int)            # 6; range: [0,inf]
    parser.add_argument("--num_boost_round", type=int)

    args = parser.parse_args()
    dataset = args.nyc_taxi_parquet
    model_path = args.model
    tree_method = args.tree_method
    learning_rate = args.learning_rate
    gamma = args.gamma
    max_depth = args. max_depth
    num_boost_round = args.num_boost_round

    params =  {   
        "verbosity": 2, 
        "tree_method": tree_method, 
        "learning_rate": learning_rate,
        "gamma": gamma,
        "max_depth": max_depth,
        "objective": "reg:squarederror"
    }
    mlflow.log_param('num_boost_round', num_boost_round)
    mlflow.log_params(params)
    cluster = LocalCluster(local_directory=tempfile.gettempdir())
    c = Client(cluster)

    df = dd.read_parquet(dataset, index=False)
    print("Loaded from parquet")
    print(df.dtypes)

    # drop redundat cols -- also xgb cannot handle dates
    drop_cols = ["tip_amount", "pickup_datetime", "dropoff_datetime", "diff"]
    df = df.drop(drop_cols, axis=1)
    train, test = df.random_split([0.8, 0.2])

    train_target = train.fare_amount
    test_target = test.fare_amount

    train_data = train.drop("fare_amount", axis=1)
    test_data = test.drop("fare_amount", axis=1)
    print("Before Training")
    print(train_data.dtypes)

    print("Crating DaskDMatrix")
    dtrain = xgb.dask.DaskDMatrix(c, train_data, train_target)
    dtest = xgb.dask.DaskDMatrix(c, test_data, test_target)

    print("Start Training")
    model = xgb.dask.train(
        c,
        params,
        dtrain,
        num_boost_round=int(num_boost_round),
        evals=[(dtest, "test")],
    )

    print("model", model)

    for metric in model['history']['test']['rmse']:
        mlflow.log_metric('test-rmse', metric)

    tempdir = tempfile.gettempdir() + '/' + str(uuid.uuid4())
    mlflow.xgboost.save_model(model['booster'], tempdir)
    shutil.copytree(tempdir, model_path, dirs_exist_ok=True)