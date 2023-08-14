# Elastic Training for AzureML: reduce cost and guard against failures

Introduction
Distributed training of machine learning models is becoming increasingly important due to the need to handle large datasets, achieve faster training times, and optimize resource utilization. Elastic Training on AzureML aims to provide support for elastic, fault-tolerant, and resilient distributed training, leveraging frameworks like PyTorch, Horovod, and Dask. In the first phase of the project, we are focusing on PyTorch, with plans to extend to other frameworks in the future. This article showcases the benefits of using elastic training in a large language model distributed training job.

### Use Cases
Elastic training can enhance several use cases:
- **Sharing Subscription Quota**: Elastic training allows a training job to start as soon as a specified minimum number of worker nodes are available, rather than waiting for the maximum number of nodes.
- **Low-Priority Compute**: In the event of preemption, elastic training allows jobs to continue with fewer nodes, preventing complete job failure or pause.
- **Fault Tolerance**: Elastic training enables the job to continue with fewer nodes in case of a hardware failure, rescaling once the node recovers.


## Comparing an AzureML Distributed Job with Elastic Training vs without

The example used in this comparison is a T5 model finetuning example that uses distributed training. The example can be found [here](https://github.com/microsoft/onnxruntime-training-examples/tree/master/T5). Both jobs are trained for 1000 epochs on the CNN_daily dataset and use the same configurations.

#### Job 1: Base
- Compute: 3 Standard_NC24rs_v3 nodes. 
- Time: 2 hours

#### Job 2: Elastic Job

The elastic job uses the same settings as the base job, but has simulated pre-emption of some of the nodes to showcase realistic use of low-priority nodes. The amount of nodes present in the job throughout training can be seen below.

<img src="https://github.com/Azure/azureml-examples/assets/73311224/36525896-67d5-44ec-b769-c7fe42ab4d14" alt="World size of elastic job throughout training." width="600"/>

By restarting the training portion of the job automatically using model checkpointing, the job is able to continue training unaffected by node pre-emption. As long as one node is still available for training, the job will not fail.

- Compute: 3 Standard_NC24rs_v3 nodes, low-priority enabled. 
- Time: 2.37 hours
- Elastic Training enabled


> NOTE: Low-Priority nodes allow you to use excess unutilized capacity for azure machine learning jobs. See [here](https://learn.microsoft.com/en-us/azure/machine-learning/how-to-manage-optimize-cost?view=azureml-api-2) for more information on low-priority nodes. 

### Cost

| Job Type                  | Compute cost per hour | # Nodes | Training Time (hours)| Total Cost |
|---------------------------|-----------------------|---------|----------------------|------------|
| Standard Configuration    |           $13.46      |    3    |       2              |  $80.76    |
| Low Priority with Elastic |           $2.69       |    3    |       2.37           |  $19.13    |

### Performance
Both jobs ended with similar performance metrics. The average loss calculated for the base training job was **0.0438**, while the average loss for the elastic training job was **0.0704**.

### Future Steps

- **Nebula Checkpointing**: Elastic Training does not currently work with Nebula Fast Checkpointing. Adding this feature so that both tools can be used together is in progress. As alternatives, Elastic Training currently works with both Huggingface and DeepSpeed default checkpointing.

- **DeepSpeed ZeRO optimizers**: Elastic Training does not work in certain scenarios with DeepSpeed ZeRO optimizers activated due complications with checkpointing. Getting these to work together is another future step.
