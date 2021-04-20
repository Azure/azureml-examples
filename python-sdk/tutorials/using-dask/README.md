---
page_type: sample
languages:
- python
- azurecli
products:
- azure-machine-learning
description: Learn how to read from cloud data and scale PyData tools (Numpy, Pandas, Scikit-Learn, etc.) with [Dask](https://dask.org) and Azure ML.
---

# Using Dask

"Dask natively scales Python" and "provides advanced parallelism for analytics, enabling performance at scale for the tools you love." It is open source, freely available, and sits in the [PyData ecosystem](https://github.com/pydata) of tools, develop in coordination with other projects like Numpy, Pandas, and Scikit-Learn. It provides familiar APIs for Python users, allows for low-level customization and streaming with a futures API, and scales up on clusters.

Dask is often compared to Spark - see [this page](https://docs.dask.org/en/latest/spark.html) to help evaluate which is the better tool for you. Common ML tools like Optuna, Scikit-Learn, XGBoost, LightGBM, and more can be distributed via Dask. There are numerous packages available for scaling on cloud clusters.

In this tutorial, the following notebooks demonstrate using Dask with Azure:

- [1.intro-to-dask.ipynb](1.intro-to-dask.ipynb)

The main [dask](https://github.com/dask/dask) and [distributed](https://github.com/dask/distributed) themselves are small and focused. Thousands of tools, some built by the Dask organization and most not, utilize Dask for parallel or distributed processing. Some of the most useful for data science include:

- [dask/adlfs](https://github.com/dask/adlfs)
- [dask/dask-ml](https://github.com/dask/dask-ml)
- [pydata/xarray](https://github.com/pydata/xarray)
- [microsoft/lightgbm](https://github.com/microsoft/lightgbm)
- [dmlc/xgboost](https://github.com/dmlc/xgboost)
- [rapidsai/cudf](https://github.com/rapidsai/cudf)
- [rapidsai/cuml](https://github.com/rapidsai/cuml)
