
We use a datastore with datapath as input file in this example. An easy way to get a datastore and datapath with some dummy data is to create a AzureML dataset. Try `az ml dataset create --file data.yml`. You will find the datastore + datapath URI in the `paths` section of the output. The format to use datastore and datapath is typically `azureml://datastores/<datastore_name>/paths/<path>/<on>/<datastore>/<filename.extn>` You can then reference that in `pipeline.yml`



Sample output for dataset creation:
```
# az ml dataset create --file data.yml 
Command group 'ml dataset' is in preview and under development. Reference and support levels: https://aka.ms/CLI_refstatus
{
  "creation_context": {
    "created_at": "2021-09-21T19:41:27.051188+00:00",
    "created_by": "Manoj Bableshwar",
    "created_by_type": "User",
    "last_modified_at": "2021-09-21T19:41:27.051188+00:00",
    "last_modified_by": "Manoj Bableshwar",
    "last_modified_by_type": "User"
  },
  "description": "sample dataset",
  "id": "/subscriptions/b8c23406-f9b5-4ccb-8a65-a8cb5dcd6a5a/resourceGroups/aesviennatesteuap/providers/Microsoft.MachineLearningServices/workspaces/testep/datasets/sampledata1234/versions/1",
  "name": "sampledata1234",
  "paths": [
    {
      "file": "azureml://datastores/workspaceblobstore/paths/LocalUpload/cec6841f346975cde1ee7d5289c5559f/data"
    }
  ],
  "resourceGroup": "aesviennatesteuap",
  "tags": {},
  "version": "1"
}
```