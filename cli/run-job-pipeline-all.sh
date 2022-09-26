
if [ -z "$1" ]
  then
    target_version="$RANDOM"
  else
    target_version=$1
fi

az ml compute create -n cpu-cluster --type amlcompute --min-instances 0 --max-instances 8 -o none
az ml compute create -n gpu-cluster --type amlcompute --min-instances 0 --max-instances 4 --size Standard_NC12 -o none
az ml data create --file assets/data/local-folder.yml --set version=$target_version -o none
az ml component create --file jobs/pipelines-with-components/basics/1b_e2e_registered_components/train.yml --set version=$target_version -o none
az ml component create --file jobs/pipelines-with-components/basics/1b_e2e_registered_components/score.yml --set version=$target_version -o none
az ml component create --file jobs/pipelines-with-components/basics/1b_e2e_registered_components/eval.yml --set version=$target_version -o none
az ml data create --file jobs/pipelines-with-components/rai_pipeline_adult_analyse/data/data_adult_test.yaml --set version=$target_version -o none
az ml data create --file jobs/pipelines-with-components/rai_pipeline_adult_analyse/data/data_adult_train.yaml --set version=$target_version -o none
az ml environment create --file jobs/pipelines-with-components/rai_pipeline_adult_analyse/environment/responsibleai-environment.yaml --set version=$target_version -o none

echo ./jobs/basics/hello-pipeline-abc.yml
bash run-job.sh ./jobs/basics/hello-pipeline-abc.yml cli_samples_v2_$target_version nowait

echo ./jobs/basics/hello-pipeline-customize-output-file.yml
bash run-job.sh ./jobs/basics/hello-pipeline-customize-output-file.yml cli_samples_v2_$target_version nowait

echo ./jobs/basics/hello-pipeline-customize-output-folder.yml
bash run-job.sh ./jobs/basics/hello-pipeline-customize-output-folder.yml cli_samples_v2_$target_version nowait

echo ./jobs/basics/hello-pipeline-default-artifacts.yml
bash run-job.sh ./jobs/basics/hello-pipeline-default-artifacts.yml cli_samples_v2_$target_version nowait

echo ./jobs/basics/hello-pipeline-io.yml
bash run-job.sh ./jobs/basics/hello-pipeline-io.yml cli_samples_v2_$target_version nowait

echo ./jobs/basics/hello-pipeline-settings.yml
bash run-job.sh ./jobs/basics/hello-pipeline-settings.yml cli_samples_v2_$target_version nowait

echo ./jobs/basics/hello-pipeline.yml
bash run-job.sh ./jobs/basics/hello-pipeline.yml cli_samples_v2_$target_version nowait

echo ./jobs/pipelines/cifar-10/pipeline.yml
bash run-job.sh ./jobs/pipelines/cifar-10/pipeline.yml cli_samples_v2_$target_version nowait

echo ./jobs/pipelines/nyc-taxi/pipeline.yml
bash run-job.sh ./jobs/pipelines/nyc-taxi/pipeline.yml cli_samples_v2_$target_version nowait

echo ./jobs/pipelines/add-column-and-word-count-using-spark/pipeline.yml
bash run-job.sh ./jobs/pipelines/add-column-and-word-count-using-spark/pipeline.yml cli_samples_v2_$target_version nowait

echo ./jobs/pipelines-with-components/basics/1a_e2e_local_components/pipeline.yml
bash run-job.sh ./jobs/pipelines-with-components/basics/1a_e2e_local_components/pipeline.yml cli_samples_v2_$target_version nowait

echo ./jobs/pipelines-with-components/basics/1b_e2e_registered_components/pipeline.yml
bash run-job.sh ./jobs/pipelines-with-components/basics/1b_e2e_registered_components/pipeline.yml cli_samples_v2_$target_version nowait

echo ./jobs/pipelines-with-components/basics/2a_basic_component/pipeline.yml
bash run-job.sh ./jobs/pipelines-with-components/basics/2a_basic_component/pipeline.yml cli_samples_v2_$target_version nowait

echo ./jobs/pipelines-with-components/basics/2b_component_with_input_output/pipeline.yml
bash run-job.sh ./jobs/pipelines-with-components/basics/2b_component_with_input_output/pipeline.yml cli_samples_v2_$target_version nowait

echo ./jobs/pipelines-with-components/basics/3a_basic_pipeline/pipeline.yml
bash run-job.sh ./jobs/pipelines-with-components/basics/3a_basic_pipeline/pipeline.yml cli_samples_v2_$target_version nowait

echo ./jobs/pipelines-with-components/basics/3b_pipeline_with_data/pipeline.yml
bash run-job.sh ./jobs/pipelines-with-components/basics/3b_pipeline_with_data/pipeline.yml cli_samples_v2_$target_version nowait

echo ./jobs/pipelines-with-components/basics/4a_local_data_input/pipeline.yml
bash run-job.sh ./jobs/pipelines-with-components/basics/4a_local_data_input/pipeline.yml cli_samples_v2_$target_version nowait

echo ./jobs/pipelines-with-components/basics/4b_datastore_datapath_uri/pipeline.yml
bash run-job.sh ./jobs/pipelines-with-components/basics/4b_datastore_datapath_uri/pipeline.yml cli_samples_v2_$target_version nowait

echo ./jobs/pipelines-with-components/basics/4c_web_url_input/pipeline.yml
bash run-job.sh ./jobs/pipelines-with-components/basics/4c_web_url_input/pipeline.yml cli_samples_v2_$target_version nowait

echo ./jobs/pipelines-with-components/basics/4d_data_input/pipeline.yml
bash run-job.sh ./jobs/pipelines-with-components/basics/4d_data_input/pipeline.yml cli_samples_v2_$target_version nowait

echo ./jobs/pipelines-with-components/basics/5a_env_public_docker_image/pipeline.yml
bash run-job.sh ./jobs/pipelines-with-components/basics/5a_env_public_docker_image/pipeline.yml cli_samples_v2_$target_version nowait

echo ./jobs/pipelines-with-components/basics/5b_env_registered/pipeline.yml
bash run-job.sh ./jobs/pipelines-with-components/basics/5b_env_registered/pipeline.yml cli_samples_v2_$target_version nowait

echo ./jobs/pipelines-with-components/basics/5c_env_conda_file/pipeline.yml
bash run-job.sh ./jobs/pipelines-with-components/basics/5c_env_conda_file/pipeline.yml cli_samples_v2_$target_version nowait

echo ./jobs/pipelines-with-components/basics/6a_tf_hello_world/pipeline.yml
bash run-job.sh ./jobs/pipelines-with-components/basics/6a_tf_hello_world/pipeline.yml cli_samples_v2_$target_version nowait

echo ./jobs/pipelines-with-components/basics/6b_pytorch_hello_world/pipeline.yml
bash run-job.sh ./jobs/pipelines-with-components/basics/6b_pytorch_hello_world/pipeline.yml cli_samples_v2_$target_version nowait

echo ./jobs/pipelines-with-components/basics/6c_r_iris/pipeline.yml
bash run-job.sh ./jobs/pipelines-with-components/basics/6c_r_iris/pipeline.yml cli_samples_v2_$target_version nowait

echo ./jobs/pipelines-with-components/image_classification_with_densenet/pipeline.yml
bash run-job.sh ./jobs/pipelines-with-components/image_classification_with_densenet/pipeline.yml cli_samples_v2_$target_version nowait

echo ./jobs/pipelines-with-components/nyc_taxi_data_regression/pipeline.yml
bash run-job.sh ./jobs/pipelines-with-components/nyc_taxi_data_regression/pipeline.yml cli_samples_v2_$target_version nowait

echo ./jobs/pipelines-with-components/pipeline_with_hyperparameter_sweep/pipeline.yml
bash run-job.sh ./jobs/pipelines-with-components/pipeline_with_hyperparameter_sweep/pipeline.yml cli_samples_v2_$target_version nowait

echo ./jobs/pipelines-with-components/rai_pipeline_adult_analyse/pipeline.yml
bash run-job.sh ./jobs/pipelines-with-components/rai_pipeline_adult_analyse/pipeline.yml cli_samples_v2_$target_version

echo ./jobs/pipelines-with-components/shakespear_sample_and_word_count_using_spark/pipeline.yml
bash run-job.sh ./jobs/pipelines-with-components/shakespear_sample_and_word_count_using_spark/pipeline.yml cli_samples_v2_$target_version nowait

az --version