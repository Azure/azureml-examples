## Working with Schedule in Azure Machine Learning CLI 2.0
This repository contains an example `YAML` file for creating `schedule` using Azure Machine learning CLI 2.0. This directory includes:

- Sample `YAML` files for creating a `schedule`. Currently the `schedule` can be used to associate with a `PipelineJob`, more job types & other resources will be supported in near future.


- To create a schedule using any of the sample YAML files end with "-schedule.yml" provided in this directory, execute following command:
```cli
> az ml schedule create -f cron-job-schedule.yml
```

- To list all the schedule from your workspace, execute following command:
```cli
> az ml schedule list -o table
```

- To quickly enable/disable an existing schedule, execute following command:
```cli
> az ml schedule enable -n schedule_name 
> az ml schedule disable -n schedule_name 
```

- To delete a schedule, execute following command, after delete a schedule, the schedule cannot trigger jobs any more, and cannot be recovered:
```cli
> az ml schedule delete -n schedule_name
```

- To update a schedule that in workspace, execute following command. Currently schedule expression, job, settings and input/output are support update:
```cli
> az ml schedule update -f cron-job-schedule.yml --set trigger.expression="15 * * * *"
```

To learn more about Azure Machine Learning CLI 2.0, [follow this link](https://docs.microsoft.com/en-us/azure/machine-learning/how-to-configure-cli).
