---
page_type: sample
languages:
- python
- azurecli
products:
- azure-machine-learning
description: Learn how to use [LightGBM](https://github.com/microsoft/lightgbm) with Azure ML.
---

# LightGBM

This tutorial demonstrates how to run LightGBM on Azure through a series of Python notebooks to demonstrate how a project might develop.

This tutorial consists of two notebooks:

- [1.local-eda.ipynb](1.local-eda.ipynb)
- [2.distributed-cpu.ipynb](2.distributed-cpu.ipynb)

The ``1.local-eda.ipynb`` notebook uses the notebook's local compute to find, read, explore, process, and train on the data. *This notebook will fail as-is if your machine is not powerful enough* - you can try working on a sample of the data (i.e. a single partition).

The code from this notebook is modified into [src/run.py](src/run.py) and the required packages in [environment.yml](environment.yml) for operationalization.

The ``2.distributed-cpu.ipynb`` notebook uses an Azure ML CPU cluster to distributed the data processing and LightGBM training steps remotely, resulting in significant speedup over standard local machines.
