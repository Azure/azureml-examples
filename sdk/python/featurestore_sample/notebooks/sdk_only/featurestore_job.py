from pyspark.sql import SparkSession
spark = SparkSession.builder.appName("AccessData").getOrCreate()


import subprocess

subprocess.run(["python", "1. Develop a feature set and register with managed feature store.py"])

subprocess.run(["python", "2. Enable materialization and backfill feature data.py"])

subprocess.run(["python", "3. Experiment and train models using features.py"])

subprocess.run(["python", "4. Enable recurrent materialization and run batch inference.py"])