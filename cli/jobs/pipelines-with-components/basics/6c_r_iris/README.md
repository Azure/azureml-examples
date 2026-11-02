---
page_type: sample
languages:
- r
products:
- azure-machine-learning
description: This sample trains an R model on the Iris dataset in an Azure Machine Learning pipeline.
---

# R Iris pipeline component

## overview

This sample trains a classification model with R on the Iris dataset using an
Azure Machine Learning pipeline component.

## objective

Show how to build a custom R environment for a non-Python Azure Machine Learning
pipeline component and log its model with MLflow.

## programming languages

R, Python, and YAML.

## estimated runtime

The pipeline runtime depends on the selected Azure Machine Learning compute.

## Custom environment

The component builds its custom image from
[docker-context/Dockerfile](docker-context/Dockerfile) using R 4.0.0. It installs
R packages from the fixed CRAN snapshot dated 2022-06-01 because newer `carrier`
releases require R 4.1 or later. The image build also loads the packages used by
the training script and MLflow helper so missing dependencies fail during image
build instead of job execution.

After changing the Dockerfile, resubmit the pipeline so Azure Machine Learning
builds and uses the updated component environment.
