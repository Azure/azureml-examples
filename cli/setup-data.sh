## IMPORTANT: this file and accompanying assets are the source for snippets in https://docs.microsoft.com/azure/machine-learning! 
## Please reach out to the Azure ML docs & samples team before before editing for the first time.

wget "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
tar -xvzf cifar-10-python.tar.gz
rm cifar-10-python.tar.gz

az ml dataset create --name cifar-10-example --version 1 --set local_path=cifar-10-batches-py
rm -r cifar-10-batches-py