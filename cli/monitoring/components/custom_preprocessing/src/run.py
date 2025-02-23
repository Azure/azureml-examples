# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Entry script for Model Data Collector Data Window Component."""

import argparse
import pandas as pd
import mltable
import tempfile
from azureml.fsspec import AzureMachineLearningFileSystem
from datetime import datetime
from dateutil import parser
from pyspark.sql import SparkSession


def init_spark():
    """Get or create spark session."""
    spark = SparkSession.builder.appName("AccessParquetFiles").getOrCreate()
    return spark


def read_mltable_in_spark(mltable_path: str):
    """Read mltable in spark."""
    spark = init_spark()
    df = spark.read.mltable(mltable_path)
    return df


def save_spark_df_as_mltable(metrics_df, folder_path: str):
    """Save spark dataframe as mltable."""
    metrics_df.write.option("output_format", "parquet").option(
        "overwrite", True
    ).mltable(folder_path)


def preprocess(
    data_window_start: str,
    data_window_end: str,
    input_data: str,
    preprocessed_input_data: str,
):
    """Extract data based on window size provided and preprocess it into MLTable.

    Args:
        production_data: The data asset on which the date filter is applied.
        monitor_current_time: The current time of the window (inclusive).
        window_size_in_days: Number of days from current time to start time window (exclusive).
        preprocessed_data: The mltable path pointing to location where the outputted mltable will be written to.
    """
    format_data = "%Y-%m-%d %H:%M:%S"
    start_datetime = parser.parse(data_window_start)
    start_datetime = datetime.strptime(
        str(start_datetime.strftime(format_data)), format_data
    )

    # TODO Validate window size
    end_datetime = parser.parse(data_window_end)
    end_datetime = datetime.strptime(
        str(end_datetime.strftime(format_data)), format_data
    )

    # Create mltable, create column with partitionFormat
    # Extract partition format
    table = mltable.from_json_lines_files(
        paths=[{"pattern": f"{input_data}**/*.jsonl"}]
    )
    # Uppercase HH for hour info
    partitionFormat = "{PartitionDate:yyyy/MM/dd/HH}/{fileName}.jsonl"
    table = table.extract_columns_from_partition_format(partitionFormat)

    # Filter on partitionFormat based on user data window
    filterStr = f"PartitionDate >= datetime({start_datetime.year}, {start_datetime.month}, {start_datetime.day}, {start_datetime.hour}) and PartitionDate <= datetime({end_datetime.year}, {end_datetime.month}, {end_datetime.day}, {end_datetime.hour})"  # noqa
    table = table.filter(filterStr)

    # Data column is a list of objects, convert it into string because spark.read_json cannot read object
    table = table.convert_column_types({"data": mltable.DataType.to_string()})

    # Use NamedTemporaryFile to create a secure temp file
    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
        save_path = temp_file.name
        table.save(save_path)

    # Save preprocessed_data MLTable to temp location
    des_path = preprocessed_input_data + "temp"
    fs = AzureMachineLearningFileSystem(des_path)
    print("MLTable path:", des_path)
    # TODO: Evaluate if we need to overwrite
    fs.upload(
        lpath=save_path,
        rpath="",
        **{"overwrite": "MERGE_WITH_OVERWRITE"},
        recursive=True,
    )

    # Read mltable from preprocessed_data
    df = read_mltable_in_spark(mltable_path=des_path)

    if df.count() == 0:
        raise Exception(
            "The window for this current run contains no data. "
            + "Please visit aka.ms/mlmonitoringhelp for more information."
        )

    # Output MLTable
    first_data_row = df.select("data").rdd.map(lambda x: x).first()

    spark = init_spark()
    data_as_df = spark.createDataFrame(pd.read_json(first_data_row["data"]))

    def tranform_df_function(iterator):
        for df in iterator:
            yield pd.concat(pd.read_json(entry) for entry in df["data"])

    transformed_df = df.select("data").mapInPandas(
        tranform_df_function, schema=data_as_df.schema
    )

    save_spark_df_as_mltable(transformed_df, preprocessed_input_data)


def run():
    """Compute data window and preprocess data from MDC."""
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_window_start", type=str)
    parser.add_argument("--data_window_end", type=str)
    parser.add_argument("--input_data", type=str)
    parser.add_argument("--preprocessed_input_data", type=str)
    args = parser.parse_args()

    preprocess(
        args.data_window_start,
        args.data_window_end,
        args.input_data,
        args.preprocessed_input_data,
    )


if __name__ == "__main__":
    run()
