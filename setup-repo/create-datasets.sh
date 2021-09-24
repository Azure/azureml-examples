# this dataset is needed for the sample under cli/jobs/pipelines-with-components/basics/4d_dataset_input
az ml dataset create -f ../cli/jobs/pipelines-with-components/basics/4d_dataset_input/data.yml

# <download_untar_cifar>
wget "https://azuremlexamples.blob.core.windows.net/datasets/cifar-10-python.tar.gz"
tar -xvzf cifar-10-python.tar.gz
# </download_untar_cifar>

# <create_cifar>
az ml dataset create --name cifar-10-example --version 1 --set local_path=cifar-10-batches-py
# </create_cifar>

# <cleanup_cifar>
rm cifar-10-python.tar.gz
rm -r cifar-10-batches-py
# </cleanup_cifar>
