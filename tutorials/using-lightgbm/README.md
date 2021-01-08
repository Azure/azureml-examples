# LightGBM

description: learn how to use [LightGBM](https://github.com/microsoft/lightgbm) distributed on Azure

This tutorial demonstrates how to run LightGBM, distributed on Azure.

The tutorial consists of two notebooks:

- [0.Untitled.ipynb](0.Untitled.ipynb)
- [1.distributed-cpu.ipynb](1.distributed-cpu.ipynb)

The first goes through the process of reading, loading, and training a model on the data locally. *This notebook is not automatically tested as it requires a somewhat large local machine to pass.* The code from this notebook is modified into [src/run.py](src/run.py) where it is used in the second notebook and distributed across many CPU nodes, resulting in much quicker execution time.
