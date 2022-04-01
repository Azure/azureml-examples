target_version="$RANDOM"

cd jobs/pipelines-with-components/basics/1a_e2e_local_components
pwd
az ml job create --file pipeline.yml
cd ../../../../

cd jobs/pipelines-with-components/basics/1b_e2e_registered_components
pwd
az ml component create --file train.yml --set version=$target_version
az ml component create --file score.yml --set version=$target_version
az ml component create --file eval.yml --set version=$target_version
az ml job create --file pipeline.yml
az ml job create --file pipeline.yml --set jobs.train.compute=azureml:gpu-cluster
cd ../../../../

cd jobs/pipelines-with-components/basics/2a_basic_component
pwd
az ml job create --file pipeline.yml
cd ../../../../

cd jobs/pipelines-with-components/basics/2b_component_with_input_output
pwd
az ml job create --file pipeline.yml
cd ../../../../

cd jobs/pipelines-with-components/basics/3a_basic_pipeline
pwd
az ml job create --file pipeline.yml
cd ../../../../

cd jobs/pipelines-with-components/basics/3b_pipeline_with_data
pwd
az ml job create --file pipeline.yml
cd ../../../../

cd jobs/pipelines-with-components/basics/4a_local_data_input
pwd
az ml job create --file pipeline.yml
cd ../../../../

cd jobs/pipelines-with-components/basics/4b_datastore_datapath_uri
pwd
az ml job create --file pipeline.yml
cd ../../../../

# cd jobs/pipelines-with-components/basics/4c_dataset_input
# az ml data create --file data.yml --version $target_version
# az ml job create --file pipeline.yml
# cd ../../../../

cd jobs/pipelines-with-components/basics/4c_web_url_input
pwd
az ml job create --file pipeline.yml
cd ../../../../

cd jobs/pipelines-with-components/basics/5a_env_public_docker_image
pwd
az ml job create --file pipeline.yml
cd ../../../../

cd jobs/pipelines-with-components/basics/5b_env_registered
pwd
az ml job create --file pipeline.yml
cd ../../../../

cd jobs/pipelines-with-components/basics/5c_env_conda_file
pwd
az ml job create --file pipeline.yml
cd ../../../../

cd jobs/pipelines-with-components/basics/6a_tf_hello_world
pwd
az ml job create --file pipeline.yml
cd ../../../../

cd jobs/pipelines-with-components/basics/6b_pytorch_hello_world
pwd
az ml job create --file pipeline.yml
cd ../../../../

cd jobs/pipelines-with-components/basics/6c_r_iris
pwd
az ml job create --file pipeline.yml
cd ../../../../

cd jobs/pipelines-with-components/image_classification_with_densenet
pwd
az ml job create --file pipeline.yml
cd ../../../

cd jobs/pipelines-with-components/nyc_taxi_data_regression
pwd
az ml job create --file pipeline.yml
cd ../../../

cd jobs/pipelines/nyc-taxi
pwd
az ml job create --file pipeline.yml
cd ../../../

cd jobs/pipelines/cifar-10
pwd
az ml job create --file pipeline.yml --web
cd ../../../

az --version
