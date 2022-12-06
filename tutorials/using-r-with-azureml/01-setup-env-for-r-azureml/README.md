
* create an Azure Machine Learning workspace if you do not already have one. You can use an existing workspace


* edit the set-environment-vars.sh file with the following information from your new or existing workspace:
    - The subscription id where the workspace is
    - The resource group name where the workspace is
    - The name of the Azure Machine Learning workspace
     
* create a compute instance if you do not already have one

* Run the ci-setup-interactive script to install additional R and Python pacakges
    - open a terminal
    - git clone this repo
    - change to 01-...
    - run bash ci-setup-interactive

* Create a data asset
    - run the create-data-asset.sh script which creates a data asset called orange-juice
