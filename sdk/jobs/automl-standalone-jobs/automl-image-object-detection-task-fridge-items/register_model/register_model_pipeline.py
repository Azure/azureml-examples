
import argparse
import json
import os
import re
import mlflow
from azureml.core import Model
from azureml.core.run import Run, _OfflineRun
from azureml.core import Workspace

parser = argparse.ArgumentParser()
parser.add_argument("--automl_best_run", help="Job Name for AutoML step")
parser.add_argument('--primary_metric', help = "Primary metric for the run")
parser.add_argument('--model_name', help = "Automl image model name to register")
parser.add_argument('--training_mltable', help = "Training mltable")
parser.add_argument('--register_always', default=False, type=lambda x: (str(x).lower() == 'true'))
args = parser.parse_args()

print("Argument 1(automl_best_run): %s" % args.automl_best_run)
print("Argument 2(primary_metric): %s" % args.primary_metric)
print("Argument 3(model_name): %s" % args.model_name)
print("Argument 3(training_mltable): %s" % args.training_mltable)
print("Argument 5(register_always): %s" % args.register_always)


# load the mlflow model
model = mlflow.pyfunc.load_model(args.automl_best_run)

# Fetch the primary metric from mlflow best run
from mlflow.tracking.client import MlflowClient
mlflow_client = MlflowClient()
best_run = mlflow_client.get_run(model.metadata.run_id)

# converting primary metric to snake case
metric_name = re.sub(r'(?<!^)(?=[A-Z])', '_', args.primary_metric).lower()
metric_value = best_run.data.metrics[metric_name]
#creating a metric tag
tags = {metric_name: metric_value}


# Fetch model from mlflow artifatcts
local_dir = "./artifact_downloads"
if not os.path.exists(local_dir):
    os.mkdir(local_dir)

# Download run's artifacts/outputs
local_path = mlflow_client.download_artifacts(
    model.metadata.run_id, "outputs", local_dir
)
model_path =  local_path + "/model.pt"

# register the model
run = Run.get_context()
ws = None
if type(run) == _OfflineRun:
    ws = Workspace.from_config()
else:
    ws = run.experiment.workspace

try:
    model = Model(ws, args.model_name)
    last_train_time = model.created_time
    print("Model with name {0} already exists.".format(args.model_name))
    print("Model was last trained on {0}.".format(last_train_time))
except Exception as e:
    print("No model already existing with name {0}".format(args.model_name))
    model = None

if model is None or args.register_always:
    if args.register_always:
        print("register_always switch is On. Proceeding to register the model..")
    # Register the model with the training MLTable and the metrics tag. 
    print("registering the model")
    model = Model.register(workspace=ws, 
                           model_path=model_path,
                           model_name=args.model_name,
                           tags=tags,
                           properties = {'training_mltable': args.training_mltable}
                         )
    print("Registered version {0} of model {1}".format(model.version, model.name))
else:
    # Retrieve the metrics of the existing model if any.
    current_metric_val = 0
    if model.tags:
        if  metric_name in model.tags:
            current_metric_val = model.tags[metric_name]
        if metric_value > float(current_metric_val):
            print("New model has a {0} value of {1} which is better than the existing model's value of {2}".format(metric_name, metric_value, current_metric_val))
            print("Registering the model..")
            # Register the model with the training MLTable and the metrics tag. 
            model = Model.register(workspace=ws,
                                   model_path=model_path,
                                   model_name=args.model_name,
                                   tags=tags,
                                   properties = {'training_mltable': args.training_mltable}
                                  )
            print("Registered version {0} of model {1} with {2}:{3}".format(model.version, model.name, metric_name, metric_value))
        else:
            # Do not register the model as its not better than the current model.
            print("Current model has a {0} value of {1} which is better than or same as the new model's value of {2}".format(metric_name, current_metric_val, metric_value))
            print("No model registered from this run.")
    else:
        print('No metrics found for the existing model. Registering model with metrics..')
        # Register the model with the training MLTable and the metrics tag. 
        model = Model.register(workspace=ws, 
                               model_path=model_path,
                               model_name=args.model_name,
                               tags=tags,
                               properties = {'training_mltable': args.training_mltable}
                               )
        print("Registered version {0} of model {1} with {2}:{3}".format(model.version, model.name, metric_name, metric_value))