---
page_type: sample
languages:
- azurecli
- python
products:
- azure-machine-learning
description: This sample shows how to run a distributed Dask job on an Azure ML compute cluster using MPI. The 24GB NYC Taxi dataset is read in CSV format by a 4 node Dask cluster, processed, and written to job output in Parquet format. 
---

# Running a Dask job

This sample shows how to run a Dask data preparation task on an Azure ML Compute Cluster, using Dask-MPI. This method leverages the [Azure ML MPI support](https://learn.microsoft.com/en-us/azure/machine-learning/how-to-train-distributed-gpu#mpi) and does not require any custom scripts or additional libraries.

To illustrate Dask usage, the 24 GB NYC Taxi dataset is read in CSV format by a 4-node Dask cluster, processed, and then written as a job output in Parquet format. Because the script uses Dask Dataframes, the compute tasks are distributed across all 4 nodes.

The file `conda.yml` contains a Conda environment definition, with all the required dependencies to run the sample script.

The minimal required dependencies to run a Dask job using MPI are the following:

- `dask`
- `dask_mpi`
- `mpi4py`

The provided environment includes other dependencies that are only useful for this sample script.

The included `job.yml` contains an Azure ML job definition to execute the script `prep_nyctaxi.py`.

The important part of that file regarding Dask is the following section:

```yaml
distribution:
  type: mpi
  process_count_per_instance: 4
resources:
  instance_count: 4
```

This is where we request to run the script using an MPI cluster of 4 instances (`instance_count`) and 4 processes per instance (`process_count_per_instance`). You should adjust these numbers according to the configuration of your cluster.

The job also defines inputs and outputs, both mounted directly from Blob Storage to the compute nodes. This means the inputs and outputs will appear on all the nodes as local folders.

The base Docker image specified in the job definition contains the necessary [OpenMPI](https://www.open-mpi.org/) libraries. You can see a full list of available base images in the GitHub repo [AzureML-Containers](https://github.com/Azure/AzureML-Containers).

## Accessing the Dask dashboard

The Dask dashboard is very useful to understand what is going on in the cluster. This sample shows a way of accessing the dashboard using SSH tunnels.

- Once the script is running, you can find the job in the Azure ML Jobs list.
- Click on the job name to open the job page.
- Click on "Outputs + logs" to access the logs of the job.
- Open the `user_logs` directory. You will see one log per MPI process, in the form `std_log_process_xx.txt`.
- Open the log named `std_log_process_01.txt`, this is where you will find the logs written by the script running on the MPI process of rank 1.
- In this log you will see a line like this: `Dask dashboard on 10.0.0.8:8787`; this gives you the internal IP address of the host where the Dask dashboard is running.

Now you need to open an SSH tunnel between your workstation and the host, so that you can access the dashboard. To do that, you will find the public IP address of your cluster, and use it to open the tunnel.

- In the Azure ML Studio, go to the "Compute" page.
- Click on "Compute Clusters".
- Click on your cluster name, for example `dask-cluster`.
- Click on Nodes. You will see a list of nodes in your cluster. Each node has a "Connection string" value with a clipboard icon. Click on the clipboard icon of any line to get the SSH command to connect to the cluster. It will look like `ssh azureuser@20.a.b.c -p 50003`.

To create the SSH tunnel, use the `-L` argument to indicate that you want to forward the connection from local port 8787 to the remote port, using the information from the logs. The final command should look like this:

```sh
ssh azureuser@20.a.b.c -p 50003 -L 8787:10.0.0.8:8787
```

Run that command, and the tunnel should be established. Connect to `http://localhost:8787/status` with your browser to access the dashboard.

## How it works

The following two lines are enough to set up the Dask cluster over MPI:

```python
# Initialize Dask over MPI
dask_mpi.initialize()
c = Client()
```

This will automatically run the Dask Scheduler on the MPI process with rank 0, the client code on rank 1, and the Dask Workers on the remaining ranks. This means that out of all the distributed processes you requested in your Azure ML job, two are used to coordinate the cluster, and the others to actually perform the compute tasks.

You can read more in the Dask-MPI documentation: [Dask-MPI with Batch Jobs](https://mpi.dask.org/en/latest/batch.html) and [How Dask-MPI Works](https://mpi.dask.org/en/latest/howitworks.html).
