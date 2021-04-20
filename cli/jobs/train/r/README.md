# R on Azure ML 2.0

This directory provides some sample projects to run R on Azure ML 2.0.

## Environment Set-up
All the projects in this directory use the `rocker/tidyverse:4.0.0-ubuntu18.04` image from Docker Hub. This image has the tidyverse and its dependencies installed (see [Rocker](https://github.com/rocker-org/rocker) for more details).

To register this environment as an asset in Azure ML, run the following commands in your terminal:

```bash
cd examples/r
az ml environment create --file r_env.yml
```

## Examples

* [Train a model](./basic-train-model/README.md) - Trains an `rpart` model on the Iris dataset. However, rather than this data being part of the package we articulate how to register a csv file as a data asset in Azure ML. The job consumes this csv file during training in the cloud.
* [Train an car accident prediction model](./accident-prediction) - Trains an accident prediction model using the `glm()` function on car accident data.
