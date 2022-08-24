# run classification experiment in v1 and register custom and mlflow model

#imports
import pickle
import mlflow
from azureml.core import Model
from azureml.core.dataset import Dataset
from azureml.core.workspace import Workspace
from azureml.train.automl import AutoMLConfig
from azureml.train.automl.run import AutoMLRun
from azureml.core.experiment import Experiment
from azureml.train.automl.constants import Tasks
from azureml.automl.core.shared.constants import Metric
from azureml.automl.core.constants import FeaturizationConfigMode
# to parametrize
workspace_name = "chpirillmaster" 
resource_group = "chpirill"
subscription_id = "ea4faa5b-5e44-4236-91f6-5483d5b17d14"
    
ws = Workspace(workspace_name=workspace_name, resource_group=resource_group, subscription_id=subscription_id)

experiment_name = 'v1-model-registration-experiment'

experiment = Experiment(ws, experiment_name)

run_id = "AutoML_009c7ebb-e4d5-41b3-be4a-51227fbd0747"
remote_run =  AutoMLRun(experiment, run_id = run_id)
print("V1 Run: ", remote_run)

# wait till the run completes
remote_run.wait_for_completion(show_output=False)

# get created custom model
best_run, fitted_model = remote_run.get_output()

#save to 
filename = 'myv1model/custom_model_created_from_v1'
pickle.dump(fitted_model, open(filename, 'wb'))

# register v1 custom model file
custom_model = Model.register(
    model_path=filename,
    model_name="custom_model_created_from_v1_file",
    description="command job custom model input v1",
    workspace=ws
    )

# register v1 custom model folder
custom_model = Model.register(
    model_path='myv1model',
    model_name="custom_model_created_from_v1_folder",
    description="command job custom model input v1",
    workspace=ws
    )

#load mlflow model
mlflow.set_tracking_uri(ws.get_mlflow_tracking_uri())

mlflow_model_path = "outputs/mlflow-model/"
uri = "runs:/" + best_run.id + "/" + mlflow_model_path

mlflow.sklearn.load_model(uri, "./")

# register v1 mlflow model
mlflow_model = Model.register(
    model_path=mlflow_model_path,
    model_name="mlflow_model_created_from_v1",
    description="command job mlflow model input v1",
    workspace=ws
    )