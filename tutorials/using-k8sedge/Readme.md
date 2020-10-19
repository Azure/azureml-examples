# Using models on Edge device with Kubernetes.

This section guides you through the steps needed to deploy a model for inferencing in GPU-enabled Kubernetes cluster on Edge device.

## Prerequisites

1. You should be able to complete this step on AS Edge https://docs.microsoft.com/en-us/azure/databox-online/azure-stack-edge-j-series-deploy-stateless-application-kubernetes
2. Have a valid Microsoft Azure subscription
3. Be able to provision GPU-enabled VMs
4. Have access to VM image repository (DockerHub account, or ACR)

## Getting Started

If you already have a VM, you need to be able to validate you have access to GPUs. See [Checking GPU availabilty](checking_gpu_availability.md) for details.

Some of this tutorial could also be done with a simplified cluster. See [Creating one-node Kubernetes cluster](one_node_k8s.md) if needed.

## Deploying on Kubernetes on Edge device

You can run the provided notebook step by step, and follow the [Creating and using model](creating_and_using_model.md) if you have any questions.

If you face problems, contact your Azure administrator for the credentials and addresses of your ACR and other network information. 

## Links

- https://docs.nvidia.com/datacenter/kubernetes/kubernetes-upstream/index.html#kubernetes-run-a-workload - NVIDIA webpage.
- https://github.com/NVIDIA/k8s-device-plugin/blob/examples/workloads/pod.yml - NVIDIA example repository.
- https://docs.microsoft.com/en-us/azure/container-registry/container-registry-get-started-docker-cli - ACR information.
- https://kubernetes.io/docs/tasks/configure-pod-container/pull-image-private-registry/ - working with private repositories in Kubernetes
