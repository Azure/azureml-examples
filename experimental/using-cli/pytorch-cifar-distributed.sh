## IMPORTANT: this file and accompanying assets are the source for snippets in https://docs.microsoft.com/azure/machine-learning! 
## Please reach out to the Azure ML docs & samples team before before editing for the first time.

# <create_compute>
az ml compute create -n gpu-cluster --min-instances 0 --max-instances 2 --type AmlCompute --size Standard_NC12
# </create_compute>

# <download_cifar>
wget "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
# </download_cifar>

# <untar_cifar>
tar -xvzf cifar-10-python.tar.gz
# </untar_cifar>

# <remove_tar>
rm cifar-10-python.tar.gz
# </remove_tar>

# <create_data>
az ml data create -n cifar-10-upload -v 1 --set local_path=cifar-10-batches-py
# </create_data>

# <create_basic_job>
job_id=`az ml job create -f jobs/train/pytorch/cifar-distributed/basic.yml -o json --query name`
# </create a basic job>

 