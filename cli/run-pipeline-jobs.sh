cd jobs/pipelines-with-components/basics/1a_e2e_local_components
az ml job create --file pipeline.yml --web
cd ../../../../

cd jobs/pipelines-with-components/basics/1b_e2e_registered_components
target_version="$RANDOM"
az ml component create --file train.yml --set version=$target_version
az ml component create --file score.yml --set version=$target_version
az ml component create --file eval.yml --set version=$target_version
az ml job create --file pipeline.yml --web
az ml job create --file pipeline.yml --set jobs.train.compute=azureml:gpu-cluster --web
cd ../../../../

cd jobs/pipelines-with-components/basics/2a_basic_component
az ml job create --file pipeline.yml --web
cd ../../../../

cd jobs/pipelines-with-components/basics/2b_component_with_input_output
az ml job create --file pipeline.yml --web
cd ../../../../

cd jobs/pipelines-with-components/basics/3a_basic_pipeline
az ml job create --file pipeline.yml --web
cd ../../../../

cd jobs/pipelines-with-components/basics/3b_pipeline_with_data
az ml job create --file pipeline.yml --web
cd ../../../../

cd jobs/pipelines-with-components/basics/4a_local_data_input
az ml job create --file pipeline.yml --web
cd ../../../../

cd jobs/pipelines-with-components/basics/4b_datastore_datapath_uri
az ml job create --file pipeline.yml --web
cd ../../../../

cd jobs/pipelines-with-components/basics/4c_dataset_input
az ml data create --file data.yml
az ml job create --file pipeline.yml --web
cd ../../../../

cd jobs/pipelines-with-components/basics/4d_web_url_input
az ml job create --file pipeline.yml --web
cd ../../../../

cd jobs/pipelines-with-components/basics/5a_env_public_docker_image
az ml job create --file pipeline.yml --web
cd ../../../../

cd jobs/pipelines-with-components/basics/5b_env_registered
az ml job create --file pipeline.yml --web
cd ../../../../

cd jobs/pipelines-with-components/basics/5c_env_conda_file
az ml job create --file pipeline.yml --web
cd ../../../../

cd jobs/pipelines-with-components/basics/6a_tf_hello_world
az ml job create --file pipeline.yml --web
cd ../../../../

cd jobs/pipelines-with-components/basics/6b_pytorch_hello_world
az ml job create --file pipeline.yml --web
cd ../../../../

cd jobs/pipelines-with-components/basics/6c_r_iris
az ml job create --file pipeline.yml --web
cd ../../../../

cd jobs/pipelines-with-components/image_classification_with_densenet
az ml job create --file pipeline.yml --web
cd ../../../

cd jobs/pipelines-with-components/nyc_taxi_data_regression
az ml job create --file pipeline.yml --web
cd ../../../

cd jobs/pipelines/nyc-taxi
az ml job create --file pipeline.yml --web
cd ../../../

cd jobs/pipelines/cifar-10
az ml job create --file pipeline.yml --web
cd ../../../
