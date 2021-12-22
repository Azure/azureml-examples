# Automated Resource Provisioning for Contoso

## Contents

This document proposes a procedure to automatically provision the resources required for running Contoso Federated Learning experiments.

## Requirements

For running Federated Learning experiments, the Contoso team needs the following ingredients:

- an "orchestrator" Azure ML workspace;
- some Kubernetes (K8s) clusters;
- connections between the K8s clusters and the Azure ML workspace;
- <other ingredients related to data store, to be added later>...

The procedure outlined in this document will provision resources that meet the requirements above.

## Prerequisites

Taken from [here](https://github.com/Azure/AML-Kubernetes#prerequisites) (along with the K8s cluster creation/connection steps).

0. Have PowerShell (or PowerShell Core if not on Windows).
1. Have access to an Azure subscription.
2. Install the [latest release of Helm 3](https://helm.sh/docs/intro/install/) - for Windows, we recommend going the _chocolatey_ route.
3. Meet the pre-requisites listed under the [generic cluster extensions documentation](https://docs.microsoft.com/azure/azure-arc/kubernetes/extensions#prerequisites).
    - Azure CLI version >=2.24.0
    - Azure CLI extension k8s-extension version >=1.0.0.
4. Install and setup the [latest AzureML CLI v2](https://docs.microsoft.com/azure/machine-learning/how-to-configure-cli).
5. Install the [Bicep CLI](https://docs.microsoft.com/en-us/azure/azure-resource-manager/bicep/install) and the [Bicep extension in VS Code](https://marketplace.visualstudio.com/items?itemName=ms-azuretools.vscode-bicep).

## Procedure

> Set `accelerators/federated-learning/automated_provisioning` as your working directory!

The procedure is fairly simple (detailed instructions are given in the below sections).
1. After setting your working directory and  verifying you are meeting the prerequisites, you will first run the `CreateK8sCluster.ps1` script (with the appropriate arguments) and log in when prompted.
2. Then you will run the `ConnectSiloToOrchestrator.ps1` script (with the appropriate arguments) and log in when prompted.

We also suggest a way to run a simple validation job (highly recommended, to make sure the newly provisioned resources are usable). That being said, there is nothing special about this job, except for the fact that it will run on the new Arc-enabled K8s cluster; if you feel more comfortable using one of your own jobs to achieve that, it is perfectly acceptable.

### Set up a silo

For starters, you need to create a K8s cluster and the associated resource group if they don't exist already. Then you'll need to connect to this cluster and create the _kube_ config file (which will be used implicitly by the second script to point at this particular cluster). There is a script that does all that for you: `CreateK8sCluster.ps1`. It takes the following input arguments.
- `SubscriptionId`: the Id of the subscription where the silo will be created. 
- `K8sClusterName`: the name of the K8s cluster to be created (default: "cont-k8s-01"). It will live in a resource group named like the cluster, with "-rg" appended.
- `RGLocation`: the location of the K8s cluster and its corresponding resource group (default: "westus2").
- `AgentCount`: the number of agents in the K8s cluster (default: 1 - beware, this should be an _int_, not a _string_).
- `AgentVMSize`: The agent VM SKU (default: "Standard_B4ms").

The command should look something like the below (with the parameters replaced by your own values of course).

```ps
./ps/CreateK8sCluster.ps1 -SubscriptionId "Your-Silo-SubscriptionId" -K8sClusterName "Name-of-Cluster-to-Create" -RGLocation "Location-of-Cluster-to-Create" -AgentCount Number-of-Agents -AgentVMSize "VM-SKU"
```

### Connect it to the orchestrator workspace

To connect a silo to an orchestrator Azure ML workspace, the following needs to happen:
- create the Azure ML workspace if it doesn't already exist;
- connect the K8s to Azure Arc; 
- deploy the Azure ML extension on the Arc cluster;
- attach the Arc cluster to the Azure ML workspace.

Here again, there is a script that does all of that for you: `ConnectSiloToOrchestrator.ps1`. It takes the following input arguments.
- `SubscriptionId_Orchestrator`: the Id of the subscription to which the orchestrator will belong.
- `AMLWorkspaceName`: the name of the orchestrator Azure ML workspace to create, if it doesn't exist already.
- `AMLWorkspaceRGName`: the name of the orchestrator resource group to create, if it doesn't exist already.
- `AMLWorkspaceLocation`: the location of the orchestrator Azure ML workspace (default: "westus2").
- `K8sClusterName`: the name of the K8s cluster to connect to the orchestrator Azure ML workspace (default: "cont-k8s-01"). **Note that this is just used to create the name of the Arc cluster and its resource group. The K8s cluster is referenced implicitly by the kube config file that was created during the previous step.**
- `AMLComputeName`: the name of the Azure ML compute to be created (default: "cont-01-compute" - must be between 2-16 characters and only contain alphanumeric characters or dashes). **This is the compute name you will be using when submitting jobs.**

The command should look something like the below (with the parameters replaced by your own values of course).

```ps
./ps/ConnectSiloToOrchestrator.ps1 -SubscriptionId_Orchestrator "Your-Orchestrator-SubscriptionId" -AMLWorkspaceName "Your-Orchestrator-Workspace-Name" -AMLWorkspaceRGName "Your-Orchestrator-Resource-Group-Name" -AMLWorkspaceLocation "Your-Orchestrator-Location" -K8sClusterName "Name-of-K8s-Cluster-to-Connect" -AMLComputeName "AML-Compute-Name-to-Create"
```

### Add more silos
Just repeat the 2 steps above for every silo you want to create.

> You need to create a cluster, then connect it. If you first create several clusters, then try to connect them, you will run into issues. This is because the connection script implicitly uses the cluster reference from the first step. 

> If you want to have your K8s cluster and the orchestrator Azure ML workspace in different subscriptions, this is possible. Just use a different subscription in each of the 2 steps, and log in accordingly when prompted to. 

### Run a simple validation job

> This simple validation job currently just tests **one** silo. You will need top run it on every one of them.

To double check that you can actually run Azure ML jobs on the Arc Cluster, we provide all the files required for a sample job, following the example [here](https://github.com/Azure/AML-Kubernetes/blob/master/docs/simple-train-cli.md). First, you'll need to open `./sample_job/job.yml` - this is the file where the job you are going to run is defined. Adjust the compute name (the part after `compute: azureml:`) to the name of your Azure ML compute.

Then you will need to create the `mnist_test` dataset if it doesn't exist already, and submit the job. The PowerShell script `RunSampleJob.ps1` will do that for you. It takes the following arguments.
- `SubscriptionId`: the Id of the subscription to which the Azure ML orchestrator workspace belongs.
- `WorkspaceName`: the name of the Azure ML orchestrator workspace.
- `ResourceGroup`: the resource group of the Azure ML orchestrator workspace.

The command should look something like the below (with the parameters replaced by your own values of course).

```ps
./sample_job/RunSampleJob.ps1 -SubscriptionId "Your-Orchestrator-SubscriptionId" -WorkspaceName "Your-Orchestrator-Workspace-Name" -ResourceGroup "Your-Orchestrator-Resource-Group"
```

## Future work
- Add more validation on input strings if issues come up.
- Validate all silos at once.
- Provision data-related resources.
- ...

## Further reading
- The part about creating/connecting K8s clusters is based on [these instructions](https://github.com/Azure/AML-Kubernetes). A summary can also be found in [this deck](https://microsoft.sharepoint.com/:p:/t/AMLDataScience/EQSxAxYrjX1BiOh3s23GpJUB81sgQfNQJFTWCRR0T8pODg?e=6hcvRL).
