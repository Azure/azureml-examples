
1. Make sure you are in the `1b_e2e_registered_components` directory for this sample.

2. Register the Components with the AzureML workspace.

```
az ml component create --file train.yml
az ml component create --file score.yml
az ml component create --file eval.yml

```
If you are re-running samples, the version specified in the component yaml may already be registered. You can edit the component yaml to bump up the version or you can simply specify a new version using the command line.

```
az ml component create --file train.yml --set version=<version_number>
az ml component create --file score.yml --set version=<version_number>
az ml component create --file eval.yml --set version=<version_number>
```

3. Submit the Pipeline Job. 

Make sure the version of the components you registered matches with the version defined in pipeline.yml. Also, make sure the compute cluster used in pipeline.yml is the one that is actually available in your workspace. 

Submit the Pipeline Job
```
az ml job create --file pipeline.yml
```

You can also override the compute from the command line
```
az ml job create --file pipeline.yml --set settings.default_compute=<your_compute>
```

