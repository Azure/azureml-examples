# Creating one-node Kubernetes cluster

To create a simple one-node Kubernetes cluster, you can use `snap` to install `microk8s`:

    $ sudo snap install microk8s --edge --classic

Add your current user to microk8s group:

    $ sudo usermod -a -G microk8s $USER
    $ sudo chown -f -R $USER ~/.kube

You will also need to re-enter the session for the group update to take place:

    $ su - $USER

Then start it:

    $ microk8s.start --wait-ready

You need to enable its components depending on the desired configuration, for example, dns and dashboard:

    $ microk8s.enable dns storage dashboard

The most important for us is the access to gpu

    $ microk8s.enable gpu

You will be able to see the nodes:

    $ microk8s.kubectl get nodes
    NAME                STATUS   ROLES    AGE   VERSION
    sandbox-dsvm-tor4   Ready    <none>   14h   v1.19.2-34+88df35f6de9eb1

And the gpu-support information in the description of the node:

    $ microk8s.kubectl describe node sandbox-dsvm-tor4
    Capacity:
    ...
    nvidia.com/gpu:     1
    ...
    Allocatable:
    ...
    nvidia.com/gpu:     1
    ...
    Namespace                   Name                                          CPU Requests  CPU Limits  Memory Requests  Memory Limits  AGE
    ---------                   ----                                          ------------  ----------  ---------------  -------------  ---
    ...
    kube-system                 nvidia-device-plugin-daemonset-hmzbl          0 (0%)        0 (0%)      0 (0%)           0 (0%)         14h
    ...
    Allocated resources:
    Resource           Requests    Limits
    --------           --------    ------
    ...
    nvidia.com/gpu     1           1
    ...

After we installed Kubernetes, you should also be able to run NVIDIA's examples,
https://github.com/NVIDIA/k8s-device-plugin, here is a [`gpu-pod`](https://github.com/NVIDIA/k8s-device-plugin/blob/examples/workloads/pod.yml)
example if ran successfully:

    $ git clone -b examples https://github.com/NVIDIA/k8s-device-plugin.git
    $ cd k8-device-plugin/workloads
    $ kubectl create -f pod.yml

    $ kubectl exec -it gpu-pod nvidia-smi
    +-----------------------------------------------------------------------------+
    | NVIDIA-SMI 384.125                Driver Version: 384.125                   |
    |-------------------------------+----------------------+----------------------+
    | GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
    | Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
    |===============================+======================+======================|
    |   0  Tesla V100-SXM2...  On   | 00000000:00:1E.0 Off |                    0 |
    | N/A   34C    P0    20W / 300W |     10MiB / 16152MiB |      0%      Default |
    +-------------------------------+----------------------+----------------------+

    +-----------------------------------------------------------------------------+
    | Processes:                                                       GPU Memory |
    |  GPU       PID   Type   Process name                             Usage      |
    |=============================================================================|
    |  No running processes found                                                 |
    +-----------------------------------------------------------------------------+

If it does not work, please check the instructions at Nvidia's examples page, https://github.com/NVIDIA/k8s-device-plugin/blob/examples/workloads/pod.yml

For generality, we will be using `kubectl` instead of `microk8s.kubectl`, and you are encouraged to alias it to a shortcut.

---

[Back to Readme.md](Readme.md)
