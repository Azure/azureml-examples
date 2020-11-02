# Creating and using serialized model

Creating models is easy if you use Azure ML, you will need to make sure you install `azureml-sdk`:

![JupiterLab](pics/jupiter_lab_azureml-sdk.png) 

You will need your config file from Azure Machine Learing to be able to create new instances of `workspace`:

![JupiterLab](pics/conf_file_download.JPG)

If your image is deployed not on a publicly-available image registry, you will need to login with your credentials. You can
retrieve your credentials from the notebook - through your workspace `ws.subscription_id`, and use 
`ContainerRegistryManagementClient`):

    ...
    imagename= "tfgpu"
    imagelabel="1.0"
    package = Model.package(ws, [model], inference_config=inference_config,image_name=imagename, image_label=imagelabel)
    package.wait_for_creation(show_output=True)
    client = ContainerRegistryManagementClient(ws._auth,subscription_id)
    result= client.registries.list_credentials(ws.resource_group, reg_name, custom_headers=None, raw=False)

    print("ACR:", package.get_container_registry)
    print("Image:", package.location)
    print("using username \"" + result.username + "\"")
    print("using password \"" + result.passwords[0].value + "\"")
    ...

It will print out the values(which you could also see at the Portal, in your Azure ML):

    ...
    ACR: 12345678901234567890.azurecr.io
    Image: 1234567dedede1234567ceeeee.azurecr.io/tfgpu:1.0
    using username: "9876543210abcdef"
    using password: "876543210987654321abcdef"
    ...

At the Kubernetes cluster where you want this image to be available, you can create a secret to use later to connect
to your ACR:

    $ kubectl create secret docker-registry secret4acr2infer --docker-server=<your-registry-server>\
        --docker-username=<your-name> --docker-password=<your-pword> --docker-email=<your-email>

Or, alternatively**(do not do this if you already created a secret using kubectl docker-registry secret)**,
you can create the secret by explicitly logging in to your container registry, and then export:

    $ docker login 12345678901234567890.azurecr.io
    Username: c6a1e081293c442e9465100e3021da63
    Password:
    Login Succeeded

This will record the authentication token in your `~/.docker/config.json`, and you will be able to
create a Kubernetes secret to use to access your private repository **(You do not need to do this if
you already created a secret using kubectl docker-registry secret)**:

    $ kubectl create secret generic secret4acr2infer \
        --from-file=.dockerconfigjson=/home/azureuser/.docker/config.json \
        --type=kubernetes.io/dockerconfigjson

For more information, please see [Pull an Image from a Private Registry](https://kubernetes.io/docs/tasks/configure-pod-container/pull-image-private-registry/)  


In the following steps we denote the image as `1234567dedede1234567ceeeee.azurecr.io/tfgpu:1.0`, you can tag your own image adhering to the naming conventions you like.

## Creating a Deployment on an Edge Device or another Kubernetes cluster

We provide the Deployment file, `deploy_infer.yaml`:

    apiVersion: apps/v1
    kind: Deployment
    metadata:
    name: my-infer
    labels:
        app: my-infer
    spec:
    replicas: 1
    selector:
        matchLabels:
        app: my-infer
    template:
        metadata:
        labels:
            app: my-infer
        spec:
        containers:
        - name: my-infer
            image: 1234567dedede1234567ceeeee.azurecr.io/tfgpu:1.0
            ports:
            # we use only 5001, but the container exposes  EXPOSE 5001 8883 8888
            - containerPort: 5001
            - containerPort: 8883
            - containerPort: 8888
            resources:
            limits:
                nvidia.com/gpu:  1
        imagePullSecrets:
        - name: secret4acr2infer

You would need to update the image source, from your own DockerHub accout or ACR you have access to.

You can deploy this Deployment like so:

    $ kubectl create -f deploy_infer.yaml

And you can see it instantiated, with pod creating, etc.:

    $ kubeclt get deployment
    NAME       READY   UP-TO-DATE   AVAILABLE   AGE
    my-infer   1/1     1            1           1m
 
## Creating a Service

You then can expose the deployment to have access to it via a Service:

    $ kubectl expose deployment my-infer --type=LoadBalancer --name=my-service-infer

You should see the Service, and if everything is ok, in a few minutes you will have an External IP address:

    $ kubectl get service
    NAME               TYPE           CLUSTER-IP       EXTERNAL-IP   PORT(S)                                        AGE
    ...
    my-service-infer   LoadBalancer   10.152.183.221   <pending>     5001:30372/TCP,8883:32004/TCP,8888:31221/TCP   1m
    ...

## Running inference

The way our inference server setup, we need to make an http POST request to it, to port 5001.
You are free to use the utility you like(curl, Postman, etc.), we provide a Python script to do it, and to 
convert the numbers into the labels this model(ResNet50) uses.

**IMPOTANT**: In the script you need to put the address of your own server, for example, the cluster-ip from the server we created earlier:

    import requests
    #downloading labels for imagenet that resnet model was trained on
    classes_entries = requests.get("https://raw.githubusercontent.com/Lasagne/Recipes/master/examples/resnet50/imagenet_classes.txt").text.splitlines()

    test_sample = open('snowleopardgaze.jpg', 'rb').read()
    print(f"test_sample size is {len(test_sample)}")

    try:
        #scoring_uri = 'http://<replace with yout edge device ip address>:5001/score'
        scoring_uri = 'http://10.152.183.221:5001/score'

        headers = {'Content-Type': 'application/json'}
        resp = requests.post(scoring_uri, test_sample, headers=headers)

        print("Found: " + classes_entries[int(resp.text.strip("[]")) - 1] )

    except KeyError as e:
        print(str(e))


Run it like so:

    $ python runtest_infer.py
    test_sample size is 62821
    Found: snow leopard, ounce, Panthera uncia

And, it should identify objects on your image. 

## (Optional) Running a notebook as a script

To run Jupyter notebooks you need an environment. Often having a simple Python script is simpler, although you
may use some UI convenience and you need to be aware of the side effects.
Here is a test run of the demo_notebook.ipynb we used for Jupyter server demo, that we exported into a Pyton file, `demo_notebook.py`:
 
```
azureuser@k8s-master-45338567-0:~/src/notebook$ python3 demo_notebook.py
2020-09-28 16:53:26.378225: W tensorflow/stream_executor/platform/default/dso_loader.cc:59] Could not load dynamic library 'libcudart.so.10.1'; dlerror: libcudart.so.10.1: cannot open shared object file: No such file or directory
2020-09-28 16:53:26.378266: I tensorflow/stream_executor/cuda/cudart_stub.cc:29] Ignore above cudart dlerror if you do not have a GPU set up on your machine.
Downloading data from https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz
11493376/11490434 [==============================] - 0s 0us/step
2020-09-28 16:53:31.831014: W tensorflow/stream_executor/platform/default/dso_loader.cc:59] Could not load dynamic library 'libcuda.so.1'; dlerror: libcuda.so.1: cannot open shared object file: No such file or directory
2020-09-28 16:53:31.831054: W tensorflow/stream_executor/cuda/cuda_driver.cc:312] failed call to cuInit: UNKNOWN ERROR (303)
2020-09-28 16:53:31.831166: I tensorflow/stream_executor/cuda/cuda_diagnostics.cc:156] kernel driver does not appear to be running on this host (k8s-master-45338567-0): /proc/driver/nvidia/version does not exist
2020-09-28 16:53:31.831468: I tensorflow/core/platform/cpu_feature_guard.cc:142] This TensorFlow binary is optimized with oneAPI Deep Neural Network Library (oneDNN)to use the following CPU instructions in performance-critical operations:  AVX2 AVX512F FMA
To enable them in other operations, rebuild TensorFlow with the appropriate compiler flags.
2020-09-28 16:53:31.852320: I tensorflow/core/platform/profile_utils/cpu_utils.cc:104] CPU Frequency: 2593905000 Hz
2020-09-28 16:53:31.852512: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x487da70 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-09-28 16:53:31.852586: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
2020-09-28 16:53:33.116834: W tensorflow/core/framework/cpu_allocator_impl.cc:81] Allocation of 188160000 exceeds 10% of free system memory.
Epoch 1/5
1875/1875 [==============================] - 6s 3ms/step - accuracy: 0.9348 - loss: 0.2199
Epoch 2/5
1875/1875 [==============================] - 6s 3ms/step - accuracy: 0.9711 - loss: 0.0959
Epoch 3/5
1875/1875 [==============================] - 5s 3ms/step - accuracy: 0.9783 - loss: 0.0695
Epoch 4/5
1875/1875 [==============================] - 6s 3ms/step - accuracy: 0.9832 - loss: 0.0532
Epoch 5/5
1875/1875 [==============================] - 6s 3ms/step - accuracy: 0.9855 - loss: 0.0443
313/313 [==============================] - 0s 1ms/step - accuracy: 0.9808 - loss: 0.0674
```

You can export and run your deployment notebook similarly.

---

[Back to Readme.md](Readme.md)
