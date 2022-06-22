from dask.distributed import Client, LocalCluster
import dask.dataframe as dd
import os, uuid, sys
import argparse
import mlflow
import tempfile
import shutil
import xgboost as xgb
from xgboost.callback import EvaluationMonitor, TrainingCallback, rabit
from mlflow.models import infer_signature
import numpy as np
from typing import Tuple, Optional, Dict
from mlflow.tracking import MlflowClient

def write_freeze():
    # log pip list before doing anything else
    from pathlib import Path
    import os

    Path("./outputs").mkdir(parents=True, exist_ok=True)
    os.system("pip list > outputs/pip_list.txt")

def print_env():
    for k, v in os.environ.items():
        print(k, v)

def rmsle(predt: np.ndarray, dtrain: xgb.DMatrix) -> Tuple[str, float]:
    ''' Root mean squared log error metric.

    :math:`\sqrt{\frac{1}{N}[log(pred + 1) - log(label + 1)]^2}`
    '''
    y = dtrain.get_label()
    predt[predt < -1] = -1 + 1e-6
    elements = np.power(np.log1p(y) - np.log1p(predt), 2)
    metric = float(np.sqrt(np.sum(elements) / len(y)))

    return 'rmsle', metric

class MLFlowLogCallBack(EvaluationMonitor):
    """
        log the last metric value of each metric to mlflow
    """
    def __init__(self, mlflow_env_vars: Dict = {}, rank: int = 0, period: int = 1, show_stdv: bool = False) -> None:
        self.mlflow_env_vars = mlflow_env_vars
        super().__init__(rank, period, show_stdv)

    def after_iteration(self, model, epoch: int,
                evals_log: TrainingCallback.EvalsLog) -> bool:
        if not evals_log:
            return False

        if rabit.get_rank() == self.printer_rank:
            for data, metric in evals_log.items():
                for metric_name, log in metric.items():
                    stdv: Optional[float] = None
                    if isinstance(log[-1], tuple):
                        score = log[-1][0]
                        stdv = log[-1][1]
                    else:
                        score = log[-1]
                    
                    for k, v in self.mlflow_env_vars.items():
                        os.environ[k] = v

                    mlflow_client = MlflowClient()
                    mlflow_client.log_metric(mlflow_run_id, f"{data}-{metric_name}", score)
        return False

if __name__ == '__main__':
    write_freeze()
    #print_env()
    
    print("Command Line Parameters: ", sys.argv)

    print("MLFLOW environment variables")
    # need to remember these since logging will happen in different sub-processes
    # and the environment variables don't all carry over.
    mlflow_env_vars = {}
    for k, v in os.environ.items():
        if k.startswith("MLFLOW"):
            print(f"{k}={v}")
            mlflow_env_vars[k] = v

    parser = argparse.ArgumentParser()
    parser.add_argument("--nyc_taxi_parquet", default="~/localfiles/nyctaxi.parquet/")   # input folder
    parser.add_argument("--model", default="../data/fare_predict")              # output folder
    parser.add_argument("--tree_method", default="auto")        # auto, exact, approx, hist
    parser.add_argument("--learning_rate", type=float, default=0.3)      # 0.3; range: [0,1]
    parser.add_argument("--gamma", type=float, default=1)              # 0; range: [0,inf]
    parser.add_argument("--max_depth", type=int, default=7)            # 6; range: [0,inf]
    parser.add_argument("--num_boost_round", type=int, default=10)

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
    #mlflow.start_run()
    mlflow.log_param('num_boost_round', num_boost_round)
    mlflow.log_params(params)

    mlflow_client = MlflowClient()
    mlflow_run_id = mlflow.active_run().info.run_id
    print(f"MLFlow run id: {mlflow_run_id}")

    cluster = LocalCluster(local_directory=tempfile.gettempdir())
    c = Client(cluster)

    df = dd.read_parquet(dataset, index=False, engine="pyarrow")
    print("Loaded from parquet")
    print(df.dtypes)

    # drop redundat cols -- also xgb cannot handle dates
    # drop_cols = ["tip_amount", "pickup_datetime", "dropoff_datetime", "diff"]
    # df = df.drop(drop_cols, axis=1)
    df = df.astype("float64")
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

    print("Starting Training")
    model = xgb.dask.train(
        c,
        params,
        dtrain,
        num_boost_round=int(num_boost_round),
        evals=[(dtest, "test")],
        custom_metric=rmsle,
        callbacks=[MLFlowLogCallBack(mlflow_env_vars=mlflow_env_vars)]
    )

    print("model", model)

    #for metric in model['history']['test']['rmse']:
    #    mlflow.log_metric('test-rmse', metric)

    signature = infer_signature(model_input=train_data.head())

    tempdir = tempfile.gettempdir() + '/' + str(uuid.uuid4())
    mlflow.xgboost.save_model(model['booster'], tempdir, signature=signature)
    shutil.copytree(tempdir, model_path, dirs_exist_ok=True)