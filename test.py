# import required libraries
from azure.ml import MLClient, dsl
from azure.ml._constants import AssetTypes
from azure.ml.entities import JobInput, JobOutput, load_component
# import parallel builder function
from azure.ml.entities import parallel

# load component funcs
parent_dir = ''
train_model = load_component(
    yaml_file=parent_dir + "./train.yml"
)

# define parallel using builder function
task = parallel.

batch_inference = parallel(
  name='',
  description='',
  inputs= dict(score_input=xx, label = xx, model=xx)
  task = FunctionTask(
    inputs= '${{inputs.score_input}}',
    code='./src',
    entry_script='score.py',
    environment='azureml:my-env:1',
    args= '''
        --label ${{inputs.label}}
        --model ${{inputs.model}}
        --output ${{outputs.scored_result}}
    '''
   ),
   task = CommandTask(
       component='azureml:xx:version',
       input_data= '${{inputs.score_input}}',
   ),
   mini_batch_size= 10
)

# Construct pipeline
@dsl.pipeline(default_compute="cpu-cluster", default_datastore="workspaceblobstore")
def parallel_in_pipeline(data):
    train_model = train_model(input_data=data)
    batch_inference = batch_inference(score_input=train_model.outputs.mltable_data
      , label=train_model.outputs.label
      , model=train_model.outputs.model)
    batch_inference.mini_batch_size=5
    

pipeline = parallel_in_pipeline(
    JobInput(type=AssetTypes.URI_FOLDER, path="https://azuremlexamples.blob.core.windows.net/datasets/iris.csv")
)

# submit pipeline job
pipeline_job = ml_client.jobs.create_or_update(pipeline, experiment_name='parallel_in_pipeline')

# show pipeline job url
print(f'Job link: {pipeline_job.services["Studio"].endpoint}')