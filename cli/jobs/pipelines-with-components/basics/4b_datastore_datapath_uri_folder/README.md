
We use a datastore with datapath as input folder in this example. An easy way to get a datastore and datapath with some dummy data is to create a AzureML data asset. Try `az ml data create --file data.yml`. You will find the datastore + datapath URI in the `paths` section of the output. The format to use datastore and datapath is typically `azureml://datastores/<datastore_name>/paths/<path>/<on>/<datastore>` You can then reference that in `pipeline.yml`

Sample output for dataset creation:
```
# az ml data create --file data.yml 

Command group 'ml dataset' is in preview and under development. Reference and support levels: https://aka.ms/CLI_refstatus
{
  "creation_context": {
    "created_at": "2022-03-01T03:09:45.723978+00:00",
    "created_by": "Long Chen",
    "created_by_type": "User",
    "last_modified_at": "2022-03-01T03:09:45.723978+00:00",
    "last_modified_by": "Long Chen",
    "last_modified_by_type": "User"
  },
  "description": "sample dataset",
  "id": "azureml:/subscriptions/d128f140-94e6-4175-87a7-954b9d27db16/resourceGroups/aml-test/providers/Microsoft.MachineLearningServices/workspaces/lochen-matser-pipline/datasets/sampledata1234/versions/1",
  "name": "sampledata1234",
  "paths": [
    {
      "folder": "azureml://datastores/workspaceblobstore/paths/LocalUpload/cc4d1f81626c8537b6e99dadbfeab622/data/"
    }
  ],
  "resourceGroup": "aml-test",
  "tags": {},
  "version": "1"
}
```