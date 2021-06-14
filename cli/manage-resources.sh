## IMPORTANT: this file and accompanying assets are the source for snippets in https://docs.microsoft.com/azure/machine-learning! 
## Please reach out to the Azure ML docs & samples team before before editing for the first time.

# <create_data>
az ml data create -f assets/data/iris-url.yml
# </create_data>

# <create_environment>
az ml environment create -f assets/environment/python-ml-basic-cpu.yml
# </create_environment>

# <create_model>
az ml model create -f assets/model/lightgbm-iris.yml
# </create_model>
