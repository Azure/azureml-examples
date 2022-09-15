#---------- 1. Connect to Azure Machine Learning Workspace ----------
# import required libraries
from azure.identity import DefaultAzureCredential, InteractiveBrowserCredential
from azure.ai.ml import MLClient, Input, Output, load_component
from azure.ai.ml.dsl import pipeline
from azure.ai.ml.entities import Environment
from azure.ai.ml.constants import AssetTypes, InputOutputModes
from azure.ai.ml.parallel import parallel_run_function, RunFunction

# Configure credential
try:
    credential = DefaultAzureCredential()
    # Check if given credential can get token successfully.
    credential.get_token("https://management.azure.com/.default")
except Exception as ex:
    # Fall back to InteractiveBrowserCredential in case DefaultAzureCredential not work
    credential = InteractiveBrowserCredential()

# Get a handle to workspace
ml_client = MLClient.from_config(credential=credential)

# Retrieve an already attached Azure Machine Learning Compute.
cpu_compute_target = "cpu-cluster"
print(ml_client.compute.get(cpu_compute_target))
gpu_compute_target = "gpu-cluster"
print(ml_client.compute.get(gpu_compute_target))

#---------- 2. Define components ----------
# load component
prepare_data = load_component(path="./src/prepare_data.yml")

# parallel task to process file data
file_batch_inference = parallel_run_function(
    name="file_batch_score",
    display_name="Batch Score with File Dataset",
    description="parallel component for batch score",
    inputs=dict(
        job_data_path=Input(
            type=AssetTypes.MLTABLE,
            description="The data to be split and scored in parallel",
        )
    ),
    outputs=dict(job_output_path=Output(type=AssetTypes.MLTABLE)),
    input_data="${{inputs.job_data_path}}",
    instance_count=2,
    max_concurrency_per_instance=1,
    mini_batch_size="1",
    mini_batch_error_threshold=1,
    retry_settings=dict(max_retries=2, timeout=60),
    logging_level="DEBUG",
    task=RunFunction(
        code="./src",
        entry_script="file_batch_inference.py",
        program_arguments="--job_output_path ${{outputs.job_output_path}}",
        environment="azureml:AzureML-sklearn-0.24-ubuntu18.04-py37-cpu:1",
    ),
)

# parallel task to process tabular data
tabular_batch_inference = parallel_run_function(
    name="batch_score_with_tabular_input",
    display_name="Batch Score with Tabular Dataset",
    description="parallel component for batch score",
    inputs=dict(
        job_data_path=Input(
            type=AssetTypes.MLTABLE,
            description="The data to be split and scored in parallel",
        ),
        score_model=Input(
            type=AssetTypes.URI_FOLDER, description="The model for batch score."
        ),
    ),
    outputs=dict(job_output_path=Output(type=AssetTypes.MLTABLE)),
    input_data="${{inputs.job_data_path}}",
    instance_count=2,
    max_concurrency_per_instance=2,
    mini_batch_size="100",
    mini_batch_error_threshold=5,
    logging_level="DEBUG",
    retry_settings=dict(max_retries=2, timeout=60),
    task=RunFunction(
        code="./src",
        entry_script="tabular_batch_inference.py",
        environment=Environment(
            image="mcr.microsoft.com/azureml/openmpi3.1.2-ubuntu18.04",
            conda_file="./src/environment_parallel.yml",
        ),
        program_arguments="--model ${{inputs.score_model}}"\
                          "--job_output_path ${{outputs.job_output_path}} "\
                          "--error_threshold 5 "\
                          "--allowed_failed_percent 30 "\
                          "--task_overhead_timeout 1200 "\
                          "--progress_update_timeout 600 "\
                          "--first_task_creation_timeout 600 "\
                          "--copy_logs_to_parent True "\
                          "--resource_monitor_interva 20 ",
        append_row_to="${{outputs.job_output_path}}",
    ),
)

#--------- 3. Build pipeline ----------
@pipeline()
def parallel_in_pipeline(pipeline_job_data_path, pipeline_score_model):

    prepare_file_tabular_data = prepare_data(input_data=pipeline_job_data_path)
    # output of file & tabular data should be type MLTable
    prepare_file_tabular_data.outputs.file_output_data.type = AssetTypes.MLTABLE
    prepare_file_tabular_data.outputs.tabular_output_data.type = AssetTypes.MLTABLE

    batch_inference_with_file_data = file_batch_inference(
        job_data_path=prepare_file_tabular_data.outputs.file_output_data
    )
    # use eval_mount mode to handle file data
    batch_inference_with_file_data.inputs.job_data_path.mode = (
        InputOutputModes.EVAL_MOUNT
    )
    batch_inference_with_file_data.outputs.job_output_path.type = AssetTypes.MLTABLE

    batch_inference_with_tabular_data = tabular_batch_inference(
        job_data_path=prepare_file_tabular_data.outputs.tabular_output_data,
        score_model=pipeline_score_model,
    )
    # use direct mode to handle tabular data
    batch_inference_with_tabular_data.inputs.job_data_path.mode = (
        InputOutputModes.DIRECT
    )

    return {
        "pipeline_job_out_file": batch_inference_with_file_data.outputs.job_output_path,
        "pipeline_job_out_tabular": batch_inference_with_tabular_data.outputs.job_output_path,
    }


pipeline_job_data_path = Input(
    path="./dataset/", type=AssetTypes.MLTABLE, mode=InputOutputModes.RO_MOUNT
)
pipeline_score_model = Input(
    path="./model/", type=AssetTypes.URI_FOLDER, mode=InputOutputModes.DOWNLOAD
)
# create a pipeline
pipeline_job = parallel_in_pipeline(
    pipeline_job_data_path=pipeline_job_data_path,
    pipeline_score_model=pipeline_score_model,
)
pipeline_job.outputs.pipeline_job_out_tabular.type = AssetTypes.URI_FILE

# set pipeline level compute
pipeline_job.settings.default_compute = "cpu-cluster"

#--------- 4. Submit pipeline job ----------
pipeline_job = ml_client.jobs.create_or_update(
    pipeline_job, experiment_name="pipeline_samples"
)
pipeline_job

# wait until the job completes
ml_client.jobs.stream(pipeline_job.name)