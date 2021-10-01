# <hello_world>
az ml job create -f jobs/basics/hello-world.yml --web
# </hello_world>

# <hello_world_full>
az ml job create -f jobs/basics/hello-world-full.yml --web
# </hello_world_full>

# <hello_world_output>
az ml job create -f jobs/basics/hello-world-output.yml --web
# </hello_world_output>

run_id=$(az ml job create -f jobs/basics/hello-world-output.yml --query name)
status=$(az ml job show -n $run_id --query status -o tsv)
running=("Queued" "Starting" "Preparing" "Running" "Finalizing")
while [[ ${running[*]} =~ $status ]]
do
  sleep 8 
  status=$(az ml job show -n $run_id --query status -o tsv)
  echo $status
done

# <hello_world_output_download>
az ml job download -n $run_id
# </hello_world_output_download>
rm -r $run_id

pip install pandas
# <iris_local>
python jobs/basics/src/hello-iris.py --iris-csv https://azuremlexamples.blob.core.windows.net/datasets/iris.csv
# </iris_local>
rm -r outputs

# <iris_literal>
az ml job create -f jobs/basics/hello-iris-literal.yml --web
# </iris_literal>

# <iris_file>
az ml job create -f jobs/basics/hello-iris-file.yml --web
# </iris_file>

# <iris_folder>
az ml job create -f jobs/basics/hello-iris-folder.yml --web
# </iris_folder>

pip install mlflow azureml-mlflow
# <mlflow_local>
python jobs/basics/src/hello-mlflow.py
# </mlflow_local>
rm -r mlruns
rm helloworld.txt

# <mlflow_remote>
az ml job create -f jobs/basics/hello-mlflow.yml --web
# </mlflow_remote>

pip install scikit-learn matplotlib
# <sklearn_local>
python jobs/single-step/scikit-learn/iris/src/main.py --iris-csv https://azuremlexamples.blob.core.windows.net/datasets/iris.csv
# </sklearn_local>
rm -r mlruns

# <sklearn_remote>
az ml job create -f jobs/single-step/scikit-learn/iris/job.yml --web
# </sklearn_remote>

run_id=$(az ml job create -f jobs/single-step/scikit-learn/iris/job.yml --query name -o tsv)
status=$(az ml job show -n $run_id --query status -o tsv)
running=("Queued" "Starting" "Preparing" "Running" "Finalizing")
while [[ ${running[*]} =~ $status ]]
do
  sleep 8 
  status=$(az ml job show -n $run_id --query status -o tsv)
  echo $status
done

# TODO - fix need to download

# <sklearn_download_register_model>
az ml job download -n $run_id
az ml model create -n sklearn-iris-example -l $run_id/model/
# </sklearn_download_register_model>
rm -r $run_id

