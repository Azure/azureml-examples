## IMPORTANT: this file and accompanying assets are the source for snippets in https://docs.microsoft.com/azure/machine-learning! 
## Please reach out to the Azure ML docs & samples team before before editing for the first time.

# <lightgbm_iris_local>
az ml job create -f jobs/train/lightgbm/iris/job.yml --set compute.target=local --web --stream
# </lightgbm_iris_local>

# <create_computes>
az ml compute create -n cpu-cluster --type AmlCompute --min-instances 0 --max-instances 40
az ml compute create -n gpu-cluster --type AmlCompute --min-instances 0 --max-instances 8 --size Standard_NC12
# </create_computes>

# <lightgbm_iris>
az ml job create -f jobs/train/lightgbm/iris/job.yml --web
# </lightgbm_iris>

# <lightgbm_iris_output>
run_id=$(az ml job create -f jobs/train/lightgbm/iris/job.yml --query name -o tsv)
# </lightgbm_iris_output>

# <show_job_in_studio>
az ml job show -n $run_id --web
# </show_job_in_studio>

# <stream_job_logs_to_console>
az ml job stream -n $run_id
# </stream_job_logs_to_console>

# <check_job_status>
status=$(az ml job show -n $run_id --query status -o tsv)
echo $status
if [[ $status == "Completed" ]]
then
  echo "Job completed"
elif [[ $status == "Failed" ]]
then
  echo "Job failed"
  exit 1
else 
  echo "Job status not failed or completed"
  exit 2
fi
# </check_job_status>

# <download_outputs>
az ml job download -n $run_id --outputs
# </download_outputs>

# <lightgbm_iris_sweep>
az ml job create -f jobs/train/lightgbm/iris/job-sweep.yml --web
# </lightgbm_iris_sweep>

# <lightgbm_iris_sweep_output>
run_id=$(az ml job create -f jobs/train/lightgbm/iris/job-sweep.yml --query name -o tsv)
# </lightgbm_iris_sweep_output>

# <show_job_in_studio>
az ml job show -n $run_id --web
# </show_job_in_studio>

# <stream_job_logs_to_console>
az ml job stream -n $run_id
# </stream_job_logs_to_console>

# <check_job_status>
status=$(az ml job show -n $run_id --query status -o tsv)
echo $status
if [[ $status == "Completed" ]]
then
  echo "Job completed"
elif [[ $status == "Failed" ]]
then
  echo "Job failed"
  exit 1
else 
  echo "Job status not failed or completed"
  exit 2
fi
# </check_job_status>
 
# <download_cifar>
wget "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
tar -xvzf cifar-10-python.tar.gz
rm cifar-10-python.tar.gz
mkdir jobs/train/pytorch/cifar-distributed/data
mv cifar-10-batches-py jobs/train/pytorch/cifar-distributed/data
# </download_cifar>

# <pytorch_cifar>
az ml job create -f jobs/train/pytorch/cifar-distributed/job.yml --web
# </pytorch_cifar>

# <pytorch_cifar_output>
run_id=$(az ml job create -f jobs/train/pytorch/cifar-distributed/job.yml --query name -o tsv)
# </pytorch_cifar_output>

# <show_job_in_studio>
az ml job show -n $run_id --web
# </show_job_in_studio>

# <stream_job_logs_to_console>
az ml job stream -n $run_id
# </stream_job_logs_to_console>

# <check_job_status>
status=$(az ml job show -n $run_id --query status -o tsv)
echo $status
if [[ $status == "Completed" ]]
then
  echo "Job completed"
elif [[ $status == "Failed" ]]
then
  echo "Job failed"
  exit 1
else 
  echo "Job status not failed or completed"
  exit 2
fi
# </check_job_status>

# <download_outputs>
az ml job download -n $run_id --outputs
# </download_outputs>

# <tensorflow_mnist>
az ml job create -f jobs/train/tensorflow/mnist-distributed/job.yml --web
# </tensorflow_mnist>

# <tensorflow_mnist_output>
run_id=$(az ml job create -f jobs/train/tensorflow/mnist-distributed/job.yml --query name -o tsv)
# </tensorflow_mnist_output>

# <show_job_in_studio>
az ml job show -n $run_id --web
# </show_job_in_studio>

# <stream_job_logs_to_console>
az ml job stream -n $run_id
# </stream_job_logs_to_console>

# <check_job_status>
status=$(az ml job show -n $run_id --query status -o tsv)
echo $status
if [[ $status == "Completed" ]]
then
  echo "Job completed"
elif [[ $status == "Failed" ]]
then
  echo "Job failed"
  exit 1
else 
  echo "Job status not failed or completed"
  exit 2
fi
# </check_job_status>

# <tensorflow_mnist_horovod>
az ml job create -f jobs/train/tensorflow/mnist-distributed-horovod/job.yml --web
# </tensorflow_mnist_horovod>

# <tensorflow_mnist_horovod_output>
run_id=$(az ml job create -f jobs/train/tensorflow/mnist-distributed-horovod/job.yml --query name -o tsv)
# </tensorflow_mnist_horovod_output>

# <show_job_in_studio>
az ml job show -n $run_id --web
# </show_job_in_studio>

# <stream_job_logs_to_console>
az ml job stream -n $run_id
# </stream_job_logs_to_console>

# <check_job_status>
status=$(az ml job show -n $run_id --query status -o tsv)
echo $status
if [[ $status == "Completed" ]]
then
  echo "Job completed"
elif [[ $status == "Failed" ]]
then
  echo "Job failed"
  exit 1
else 
  echo "Job status not failed or completed"
  exit 2
fi
# </check_job_status>
