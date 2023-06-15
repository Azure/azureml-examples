# <upload_code>
az storage blob upload -c $AZUREML_DEFAULT_CONTAINER -n paths/data -f cli/jobs/spark/data/titanic.csv --account-name $AZURE_STORAGE_ACCOUNT
# </upload_code>