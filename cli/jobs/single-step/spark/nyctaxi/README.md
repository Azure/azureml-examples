# README 
# Running a Spark job

This example shows how a single node Spark job can be run on an AzureML cluster. In this example we are using 1 node using this job yaml. 

> Since the script is writing the output to the local drive, and since the dataset is cached on the same local drive, your cluster nodes need to have enough free space on the local volume to accomodate pretty much the whole input and output datasets used (to be on the safe side). The input dataset will be 47GB, the parquet output is about 4GB. Using STANDARD_D15_V2 VMs to build your cluster will give you close to 1TB of free disk space and works well even for bigger datasets.

```bash
az ml job create --file job.yml --subscription <sub-name> --resource-group <rg-name> --workspace-name <ws-name> --stream
```

## Known Issues and workarounds
* Spark Configurations in v2
* Use Synapse/HDI/Databrick Compute for Spark in v2

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

### Use Synapse/HDI/Databrick Compute for Spark in v2
Support for running Spark jobs on Synapse/HDI/Databricks in v2 is planned.