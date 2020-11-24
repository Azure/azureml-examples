# Using Dask

description: learn how to read from cloud data and scale PyData tools (Numpy, Pandas, Scikit-Learn, etc.) with [Dask](https://github.com/dask/dask)

"Dask natively scales Python" and "provides advanced parallelism for analytics, enabling performance at scale for the tools you love." It is open source, freely available, and sits in the [PyData ecosystem](https://github.com/pydata) of tools, develop in coordination with other projects like Numpy, Pandas, and Scikit-Learn. It provides familiar APIs for Python users, allows for low-level customization and streaming with a futures API, and scales up on clusters.

Dask is often compared to Spark - see [this page](https://docs.dask.org/en/latest/spark.html) to help evaluate which is the better tool for you. Common ML tools like Optuna, Scikit-Learn, XGBoost, LightGBM, and more can be distributed via Dask. There are numerous packages available for scaling on cloud clusters.

In this tutorial, the following notebooks demonstrate using Dask with Azure:

- [1.intro-to-dask.ipynb](1.intro-to-dask.ipynb)
- [2.dask-cloudprovider.ipynb](2.dask-cloudprovider.ipynb)
