## IMPORTANT: this file and accompanying assets are the source for snippets in https://docs.microsoft.com/azure/machine-learning! 
## Please reach out to the Azure ML docs & samples team before before editing for the first time.

# <download_cifar>
wget "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
# </download_cifar>

# <untar_cifar>
tar -xvzf cifar-10-python.tar.gz
# </untar_cifar>

# <remove_tar>
rm cifar-10-python.tar.gz
# </remove_tar>

# <move_data_to_project_dir>
mkdir jobs/train/pytorch/cifar-distributed/data
mv cifar-10-batches-py jobs/train/pytorch/cifar-distributed/data
# </move_data_to_project_dir>

# <create_data>
#az ml data create -n cifar-10-upload -v 2 --set local_path=cifar-10-batches-py --set path=cifar-10-batches-py
# </create_data>

# <create_basic_job>
job_id=`az ml job create -f jobs/train/pytorch/cifar-distributed/basic.yml --query name -o tsv`
# </create a basic job>

# <show_job_in_studio>
az ml job show -n $job_id --web
# </show_job_in_studio>

# <stream_job_logs_to_console>
az ml job stream -n $job_id
# </stream_job_logs_to_console>

# <check_job_status>
status=`az ml job show -n $job_id --query status -o tsv`
echo $status
if [[ $status == "Completed" ]]
then
  echo "Job completed"
elif [[ $status ==  "Failed" ]]
then
  echo "Job failed"
  exit 1
else 
  echo "Job status not failed or completed"
  exit 2
fi
# </check_job_status>

# <download_outputs>
az ml job download -n $job_id --outputs
# </download_outputs>
