# Deploy a Triton model with metrics using a custom container. 

In this example, we deploy a model using Prometheus using a custom container. We first build a docker image locally, push to the container registery, and then create an AML endpoint deployment.

Please have docker installed before running the below script: https://docs.docker.com/engine/install/

This example can be run end-to-end by executing the `deploy-custom-container-triton-metric-model.sh` script in the `CLI` directory.

# Checking the metrics

You can open up the application insights attached to your AML workspace. Then go to "monitoring" on the left side tabs, then metrics. You should be able to see them under the Metric dropwdown under "Custom"