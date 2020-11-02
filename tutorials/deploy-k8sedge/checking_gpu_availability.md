# Checking GPU availability.

You can see the hardware using `lspci`:

    $ lspci
    ...
    0001:00:00.0 3D controller: NVIDIA Corporation GK210GL [Tesla K80] (rev a1)
    ...

And you can use `nvidia-smi` utility to see the gpu driver and CUDA versions:    

    $ nvidia-smi
    Thu Sep 17 18:03:11 2020
    +-----------------------------------------------------------------------------+
    | NVIDIA-SMI 450.36.06    Driver Version: 450.36.06    CUDA Version: 11.0     |
    |-------------------------------+----------------------+----------------------+
    | GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
    | Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
    |                               |                      |               MIG M. |
    |===============================+======================+======================|
    |   0  Tesla K80           On   | 00000001:00:00.0 Off |                    0 |
    | N/A   57C    P0    59W / 149W |  10957MiB / 11441MiB |      0%      Default |
    |                               |                      |                  N/A |
    +-------------------------------+----------------------+----------------------+

    +-----------------------------------------------------------------------------+
    | Processes:                                                                  |
    |  GPU   GI   CI        PID   Type   Process name                  GPU Memory |
    |        ID   ID                                                   Usage      |
    |=============================================================================|
    |    0   N/A  N/A     19918      C   ...5b8c54360196ff/bin/python    10952MiB |
    +-----------------------------------------------------------------------------+

You also can check that your gpus are available from containers.

Please see [NVIDIA webpage](https://docs.nvidia.com/datacenter/kubernetes/kubernetes-upstream/index.html#kubernetes-run-a-workload) if you have any problems. You should see something like this, for example:

    $ sudo docker run --rm --runtime=nvidia nvidia/cuda nvidia-smi
    Unable to find image 'nvidia/cuda:latest' locally
    latest: Pulling from nvidia/cuda
    3ff22d22a855: Pull complete
    e7cb79d19722: Pull complete
    323d0d660b6a: Pull complete
    b7f616834fd0: Pull complete
    c2607e16e933: Pull complete
    46a16da628dc: Pull complete
    4871b8b75027: Pull complete
    e45235afa764: Pull complete
    250da266cf64: Pull complete
    78f4b6d02e6c: Pull complete
    ebf42dcedf4b: Pull complete
    Digest: sha256:0fe0406ec4e456ae682226751434bdd7e9b729a03067d795f9b34c978772b515
    Status: Downloaded newer image for nvidia/cuda:latest
    Thu Sep 17 17:06:27 2020
    +-----------------------------------------------------------------------------+
    | NVIDIA-SMI 418.87.01    Driver Version: 418.87.01    CUDA Version: 11.0     |
    |-------------------------------+----------------------+----------------------+
    | GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
    | Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
    |===============================+======================+======================|
    |   0  Tesla K80           On   | 0000DE85:00:00.0 Off |                    0 |
    | N/A   39C    P8    25W / 149W |      0MiB / 11441MiB |      0%      Default |
    +-------------------------------+----------------------+----------------------+

    +-----------------------------------------------------------------------------+
    | Processes:                                                       GPU Memory |
    |  GPU       PID   Type   Process name                             Usage      |
    |=============================================================================|
    |  No running processes found                                                 |
    +-----------------------------------------------------------------------------+

---

[Back to Readme.md](Readme.md)
