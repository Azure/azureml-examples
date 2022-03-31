# echo '{"subscription_id": "96aede12-2f73-41cb-b983-6d11a904839b", "resource_group": "sdk", "workspace_name": "sdk-westus2"}' > ../.azureml/config.json 

cd .\1a_pipeline_with_components_from_yaml\
papermill pipeline_with_components_from_yaml.ipynb out.ipynb -k python
cd ..

cd .\1b_pipeline_with_python_function_components\
papermill pipeline_with_python_function_components.ipynb out.ipynb -k python
cd ..

cd .\1c_pipeline_with_hyperparameter_sweep\
papermill pipeline_with_hyperparameter_sweep.ipynb out.ipynb -k python
cd ..

cd .\1d_pipeline_with_non_python_components\
papermill pipeline_with_non_python_components.ipynb out.ipynb -k python
cd ..

cd .\1e_pipeline_with_registered_components\
papermill pipeline_with_registered_components.ipynb out.ipynb -k python
cd ..

cd .\2a_train_mnist_with_tensorflow\
papermill train_mnist_with_tensorflow.ipynb out.ipynb -k python
cd ..

cd .\2b_train_cifar_10_with_pytorch\
papermill train_cifar_10_with_pytorch.ipynb out.ipynb -k python
cd ..

cd .\2c_nyc_taxi_data_regression\
papermill nyc_taxi_data_regression.ipynb out.ipynb -k python
cd ..

cd .\2d_image_classification_with_densenet\
papermill image_classification_with_densenet.ipynb out.ipynb -k python
cd ..