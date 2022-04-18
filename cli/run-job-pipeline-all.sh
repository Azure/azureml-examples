
if [ -z "$1" ]
  then
    target_version="$RANDOM"
  else
    target_version=$1
fi

python run-job-pipeline-all.py update $target_version

az ml job create --file ./jobs/pipelines/cifar-10/pipeline.yml --set experiment_name=cli_samples_v2_$target_version

az ml job create --file ./jobs/pipelines/nyc-taxi/pipeline.yml --set experiment_name=cli_samples_v2_$target_version

az ml job create --file ./jobs/pipelines-with-components/basics/1a_e2e_local_components/pipeline.yml --set experiment_name=cli_samples_v2_$target_version

az ml job create --file ./jobs/pipelines-with-components/basics/1b_e2e_registered_components/pipeline.yml --set experiment_name=cli_samples_v2_$target_version

az ml job create --file ./jobs/pipelines-with-components/basics/2a_basic_component/pipeline.yml --set experiment_name=cli_samples_v2_$target_version

az ml job create --file ./jobs/pipelines-with-components/basics/2b_component_with_input_output/pipeline.yml --set experiment_name=cli_samples_v2_$target_version

az ml job create --file ./jobs/pipelines-with-components/basics/3a_basic_pipeline/pipeline.yml --set experiment_name=cli_samples_v2_$target_version

az ml job create --file ./jobs/pipelines-with-components/basics/3b_pipeline_with_data/pipeline.yml --set experiment_name=cli_samples_v2_$target_version

az ml job create --file ./jobs/pipelines-with-components/basics/4a_local_data_input/pipeline.yml --set experiment_name=cli_samples_v2_$target_version

az ml job create --file ./jobs/pipelines-with-components/basics/4b_datastore_datapath_uri/pipeline.yml --set experiment_name=cli_samples_v2_$target_version

az ml job create --file ./jobs/pipelines-with-components/basics/4c_web_url_input/pipeline.yml --set experiment_name=cli_samples_v2_$target_version

az ml job create --file ./jobs/pipelines-with-components/basics/4d_data_input/pipeline.yml --set experiment_name=cli_samples_v2_$target_version

az ml job create --file ./jobs/pipelines-with-components/basics/5a_env_public_docker_image/pipeline.yml --set experiment_name=cli_samples_v2_$target_version

az ml job create --file ./jobs/pipelines-with-components/basics/5b_env_registered/pipeline.yml --set experiment_name=cli_samples_v2_$target_version

az ml job create --file ./jobs/pipelines-with-components/basics/5c_env_conda_file/pipeline.yml --set experiment_name=cli_samples_v2_$target_version

az ml job create --file ./jobs/pipelines-with-components/basics/6a_tf_hello_world/pipeline.yml --set experiment_name=cli_samples_v2_$target_version

az ml job create --file ./jobs/pipelines-with-components/basics/6b_pytorch_hello_world/pipeline.yml --set experiment_name=cli_samples_v2_$target_version

az ml job create --file ./jobs/pipelines-with-components/basics/6c_r_iris/pipeline.yml --set experiment_name=cli_samples_v2_$target_version

az ml job create --file ./jobs/pipelines-with-components/image_classification_with_densenet/pipeline.yml --set experiment_name=cli_samples_v2_$target_version

az ml job create --file ./jobs/pipelines-with-components/nyc_taxi_data_regression/pipeline.yml --set experiment_name=cli_samples_v2_$target_version

az ml job create --file ./jobs/pipelines-with-components/pipeline_with_hyperparameter_sweep/pipeline.yml --set experiment_name=cli_samples_v2_$target_version --web

bash run-job.sh ./jobs/pipelines-with-components/rai_pipeline_adult_analyse/pipeline.yml cli_samples_v2_$target_version

python run-job-pipeline-all.py recover

az --version