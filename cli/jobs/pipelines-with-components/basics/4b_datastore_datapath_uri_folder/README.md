
We use a datastore with datapath as input folder in this example. An easy way to get a datastore and datapath with some dummy data is to create a AzureML data asset. Try `az ml data create --file data.yml`. You will find the datastore + datapath URI in the `paths` section of the output. The format to use datastore and datapath is typically `azureml://datastores/<datastore_name>/paths/<path>/<on>/<datastore>` You can then reference that in `pipeline.yml`

Sample output for data asset creation:
```
# az ml data create --file data.yml 

Command group 'ml data' is in preview and under development. Reference and support levels: https://aka.ms/CLI_refstatus
Uploading data (0.0 MBs): 100%|#################################################################################################################################################################################################| 510/510 [00:00<00:00, 2244.91it/s]


{
  "creation_context": {
    "created_at": "2022-03-17T05:37:27.038342+00:00",
    "created_by": "Long Chen",
    "created_by_type": "User",
    "last_modified_at": "2022-03-17T05:37:27.038342+00:00",
    "last_modified_by": "Long Chen",
    "last_modified_by_type": "User"
  },
  "description": "sample dataset",
  "id": "/subscriptions/6560575d-fa06-4e7d-95fb-f962e74efd7a/resourceGroups/azureml-examples/providers/Microsoft.MachineLearningServices/workspaces/main/datasets/sampledata/versions/1",
  "name": "sampledata",
  "path": "azureml://datastores/workspaceblobstore/paths/LocalUpload/cc4d1f81626c8537b6e99dadbfeab622/data",
  "resourceGroup": "azureml-examples",
  "tags": {},
  "type": "uri_file",
  "version": "1"
}
```