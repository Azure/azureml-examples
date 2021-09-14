---
page_type: sample
languages:
- azurecli
- python
products:
- azure-machine-learning
description: This sample shows how to run a distributed Dask job on an Azure ML compute cluster. The 24GB NYC Taxi dataset is read in CSV format by a 4 node Dask cluster, processed and then written to job output in parquet format. 
---

# Running a Dask job

This example shows how a distribted Dask job can be run on multiple nodes of a cluster. In this example we are using 4 nodes using this job yaml. The startup of the cluster is done by the `startDask.py` script which launches a scheduler and a worker on the first node of the cluster and a worker on all the other nodes.

> Since the script is writing the output to the local drive, and since the dataset is cached on the same local drive, your cluster nodes need to have enough free space on the local volume to accomodate pretty much the whole input and output datasets used (to be on the safe side). The input dataset will be 24GB, the parquet output is about 4GB. Using STANDARD_D15_V2 VMs to build your cluster will give you close to 1TB of free disk space and works well even for bigger datasets.

If a --script parameter is provided, then the script will run that script after the cluster has been brought up and the job will be terminated after the script has completed. To start a Dask cluster for interactive work, don't provide a --script parameter, which will have the job run indefinitely (i.e. until you terminate it).

The job below is currently launched with `distribution: .. type: pytorch` since that gives the full flexibility of assigning the work to the different nodes of the cluster by just checking the $RANK environment variable. In the future we will provide a more generic name for that mode of launching a distributed job.

For debugging and interactive work, the script also launches a Jupyter server on the first node which can be accessed by ssh tunnelling into a node of the cluster (assuming the cluster is not in a VNet). Make sure to provide your SSH public key while setting up the cluster (in ml.azure.com Create Compute Cluster/Settings/Enable SSH access/Use existing public key). Then, on your local laptop computer, you can run a command similar to the one below:

``` bash
ssh azureuser@20.67.29.11 -p 50000 -L 9999:10.0.0.4:8888 -L 9797:10.0.0.4:8787
```

In the above example:

- `20.67.29.11` is the public IP address of any of the nodes on the cluster (find in ml.azure.com under Compute/Compute Clusters/<cluster name>/Nodes)
- `50000` is the port of that same node
- `9999` and `9797` are the ports where the jupyter server and the dask dashboard will be reachable on your local machine
- `10.0.0.4` is the private IP of the head node of the job, as will be logged by the job and show up in run history (find in ml.azure.com under Experiments/dask/<run id>/details/metrics/headnode) 

Then you should be able to access the following urls:

- `http://localhost:9999?token=<jupyter-token>` for Jupyter, where <jupyter-token> is jupyter-token that will show up in run history (find in ml.azure.com under Experiments/dask/<run id>/details/metrics/jupyter-token) 
- `http://localhost:9797` for the Dask Dashboard
