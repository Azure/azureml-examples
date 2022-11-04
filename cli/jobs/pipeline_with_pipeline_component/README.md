This is a 3 nodes sample pipeline job. There are 2 pipeline nodes in the job, each of the pipeline node uses locally defined components - train, score and eval. 

First register the pipeline (components\train\train_pipeline_component.yml) as a component in the registry.

e.g.:
```
az ml component create -n <component_name> -v <component_version> -f components\train_pipeline_component.yml --registry-name <registry_name>
```

Once this is done, get the component_id and replace those in the  pipeline.yml file in the "component:" section. Then run a pipeline job using this component 


e.g.:
```
az ml job create -n pipeline-from-ws-comp -f pipeline.yml -w <workspace_name> 
```

 to submit the pipeline job in the workspace. 
