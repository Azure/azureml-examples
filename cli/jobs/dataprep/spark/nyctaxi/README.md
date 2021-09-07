# README 

## Register the AzureML Spark Environment

Create an AML Spark environment with the following:
* Using the **AzureML-PySpark-MmlSpark-0.15** curated environemnt
* Add the 'numpy' as a conda dependency
* Add the needed Spark packages and repos

BUG:
For AzureML-PySpark-MmlSpark-0.15 we need to update the MML Spark container: mcr.microsoft.com/mmlspark/release:0.15

Run this SDK code to register
```python
from azureml.core.environment import Environment
from azureml.core import RunConfiguration
from azureml.core.conda_dependencies import CondaDependencies

spark_env=Environment.get(workspace=ws, name="AzureML-PySpark-MmlSpark-0.15")
spark_env = spark_env.clone("PySpark-MmlSpark-Alt")

# Add 
conda_dep = CondaDependencies()

# Installs numpy conda package
conda_dep.add_conda_package('numpy')
conda_dep.add_pip_package("scikit-learn")
conda_dep.add_pip_package("pandas")

# Adds dependencies to PythonSection of myenv
spark_env.python.conda_dependencies=conda_dep

### BROKEN IN V2 ###
spark_env.spark.packages = [{"group": "com.databricks","artifact": "spark-xml_2.11","version": "0.6.0"}]
spark_env.spark.repositories = ["https://mvnrepository.com/artifact/com.databricks/spark-xml"]
### BROKEN IN V2 ###

spark_env.register(ws)
```

## Coming soon
* Spark Configurations in v2
* Spark package and Spark repo support in environments in v2
* Use Synapse/HDI/Databrick Compute for Spark in v2
* Data attach as HDFS support in v2
* Pipeline support for Spark in v2
* Multi-node Spark support for AzureML compute?


### Spark Configuration in v2
This is not available for v2 in the Job spec, but you can add config to your Spark script, to dynamically load.

### v2 
Add config to your script

``` python
import pyspark
from pyspark import SparkConf, SparkContext 
spark = (
    pyspark.sql.SparkSession
    .builder
    .appName("Your App Name")
    .config("spark.executor.cores",1)
    .config("spark.executor.instances", 1)
    .config("spark.executor.memory","1g")
    .config("spark.executor.cores",1)
    .config("spark.executor.instances", 1 )
    .getOrCreate())

sc = spark.sparkContext

```
### v1 equivelent
```python
from azureml.core import RunConfiguration
from azureml.core import ScriptRunConfig 
from azureml.core import Experiment

run_config = RunConfiguration(framework="pyspark")
run_config.target = synapse_compute_name

run_config.spark.configuration["spark.driver.memory"] = "1g" 
run_config.spark.configuration["spark.driver.cores"] = 2 
run_config.spark.configuration["spark.executor.memory"] = "1g" 
run_config.spark.configuration["spark.executor.cores"] = 1 
run_config.spark.configuration["spark.executor.instances"] = 1 

run_config.environment.python.conda_dependencies = conda_dep

script_run_config = ScriptRunConfig(source_directory = './code',
                                    script= 'dataprep.py',
                                    arguments = ["--tabular_input", input1, 
                                                 "--file_input", input2,
                                                 "--output_dir", output],
                                    run_config = run_config)
```

### Spark Env support in v2 for packages and reps
This will require being able to set the run config to that pyspark framework. So the AML Environment can install need Spark packages

'run_config = RunConfiguration(framework="pyspark")'
Coming soon

### Use Synapse/HDI/Databrick Compute for Spark in v2
Coming soon