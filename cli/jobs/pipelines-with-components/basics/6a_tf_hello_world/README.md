This example doesn't run actual distributed training but shows the distributed training environment available to training scripts with the `$TF_CONFIG` environment variable. It also explains how to override distributed training settings defined in a component in a job. See `tf-mnist` for an actual training example. 

`instance_count` under `resources` defines the number of compute nodes. `worker_count`. 

TensorFlow distributed mode has two configurable properties, `worker_count` and `parameter_server_count` under the `distribution` section specified by `type: tensorflow`. `parameter_server_count` defaults to 0 unless specified. The `worker_count`, if not specified, will default to the `instance_count` under `resources`.

Fields such as `resources` and `distribution` are immutable for a specific component version once it is registered with the AzureML Workspace. You need to register a new version of the component to update them. Having said that, you can update these values with the `overrides` section for a specific instance of a `component` job. `overrides` section follows the same schema as the component for the `resources` and `distribution` sections. 

In this example, `instance_count` in the component is set to 2, but we override it to 3 in the `component` job. Similarly, we override `worker_count` from 2 in the component to 3 in the job.


