# Using models on Edge device with Kubernetes.

description: learn how to use Kubernetes on Edge device for deployment and inferencing

This section guides you through the steps needed to deploy a model for inferencing in GPU-enabled Kubernetes cluster on Edge device.

## Prerequisites

1. You should be able to complete this step on AS Edge https://docs.microsoft.com/en-us/azure/databox-online/azure-stack-edge-j-series-deploy-stateless-application-kubernetes
2. Have a valid Microsoft Azure subscription
3. Be able to provision GPU-enabled VMs
4. Have access to VM image repository (DockerHub account, or ACR)

## Getting Started, deploying on Kubernetes on Edge device

Once you have Edge device, run the provided notebook step by step, [production-deploy-to-k8s-gpu.ipynb](production-deploy-to-k8s-gpu.ipynb).

If you have any questions, please see the instructions in [Creating and using model](creating_and_using_model.md),
or contact your Azure administrator for the credentials and addresses of your ACR and other network information. 

## Links

- https://docs.nvidia.com/datacenter/kubernetes/kubernetes-upstream/index.html#kubernetes-run-a-workload - NVIDIA webpage.
- https://github.com/NVIDIA/k8s-device-plugin/blob/examples/workloads/pod.yml - NVIDIA example repository.
- https://docs.microsoft.com/en-us/azure/container-registry/container-registry-get-started-docker-cli - ACR information.
- https://kubernetes.io/docs/tasks/configure-pod-container/pull-image-private-registry/ - working with private repositories in Kubernetes
- [Creating one-node Kubernetes cluster](one_node_k8s.md)
- [Checking GPU availabilty](checking_gpu_availability.md)
