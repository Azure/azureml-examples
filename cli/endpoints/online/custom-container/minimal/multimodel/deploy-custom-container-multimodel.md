# Deploy multiple models to one deployment using a custom container 
In this example, we create a deployment that serves multiple models for inference - and do so using a custom container. A custom container is not strictly necessary for serving multiple models.

This example can be run end-to-end by following the `deploy-custom-container-multimodel.sh` script in the `CLI` directory. 

## Building the custom container
The custom container used in this example is defined in the `multimodel-minimal.dockerfile`. This dockerfile adds dependencies to the base image using a simple pip one-liner. For information on how to add conda dependencies, see the `deploy-custom-container-minimal` example. In this example, both of the models share the same Python requirements and so they can use the same base image and Python environment. A deployment may only have one environment associated with it, and thus only one image.

## Registering a multimodel model asset
To deploy multiple ML models to one deployment, the model directories (or files) should be combined into a single registered model container asset on Azure. A manifest may be added manually to contain the paths to the different models for convenience. In either case, upon initialization, the combined model container is mounted to the deployment as usual and any loading logic for multiple models is handled by the scoring script. 