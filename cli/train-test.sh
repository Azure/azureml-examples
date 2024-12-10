# <mlflow_uri>
az ml workspace show --query mlflow_tracking_uri -o tsv
# </mlflow_uri>

# <sklearn_iris>
az ml job create -f jobs/single-step/scikit-learn/iris/job.yml --web
# </sklearn_iris>

train_run_id=$(az ml job create -f jobs/single-step/scikit-learn/iris/job.yml --query name -o tsv)
if [[ -z "$train_run_id" ]]
then
  echo "Job creation failed for $train_run_id"
  exit 3
fi
status=$(az ml job show -n $train_run_id --query status -o tsv)
if [[ -z "$status" ]]
then
  echo "Status query failed"
  exit 4
fi
running=("Queued" "Starting" "Preparing" "Running" "Finalizing")
while [[ ${running[*]} =~ $status ]]
do
  sleep 8 
  status=$(az ml job show -n $train_run_id --query status -o tsv)
  echo $status
done

# </sklearn_download_register_model>

# <model_test>
az ml job create -f jobs/built-in/model-test/job.yml --web --set inputs.model_uri=runs:/$train_run_id/model
# </model_test>

test_run_id=$(az ml job create -f jobs/built-in/model-test/job.yml --query name -o tsv)
if [[ -z "$test_run_id" ]]
then
  echo "Job creation failed for $test_run_id"
  exit 3
fi
status=$(az ml job show -n $cd --query status -o tsv)
if [[ -z "$status" ]]
then
  echo "Status query failed"
  exit 4
fi
running=("Queued" "Starting" "Preparing" "Running" "Finalizing")
while [[ ${running[*]} =~ $status ]]
do
  sleep 8 
  status=$(az ml job show -n $test_run_id --query status -o tsv)
  echo $status
done
