# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Entry script for Data Drift Compute Metrics Spark Component."""

import argparse
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, StringType, FloatType
from pyspark.sql.functions import stddev


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


def _create_output_dataframe(data):
    """Get Output DataFrame Schema."""
    schema = StructType(
        [
            StructField("group", StringType(), True),
            StructField("metric_value", FloatType(), True),
            StructField("metric_name", StringType(), True),
            StructField("group_pivot", StringType(), True),
            StructField("threshold_value", FloatType(), True),
        ]
    )
    return init_spark().createDataFrame(data=data, schema=schema)


def _create_row(metric, group, group_pivot, value, threshold):
    return {
        "metric_name": metric,
        "group": group,
        "metric_value": value,
        "threshold_value": threshold,
        "group_pivot": group_pivot,
    }


def _compute_max_standard_deviation(df, std_deviation_threshold: float):
    standard_deviations = df.agg(
        *[stddev(column).alias(column) for column in df.columns]
    ).collect()[0]

    rows = []
    for feature in df.columns:

        if feature == None:
            continue

        rows.append(
            _create_row(
                metric="MaxStandardDeviation",
                group=feature,
                group_pivot="",
                value=standard_deviations[feature],
                threshold=std_deviation_threshold,
            )
        )

    return _create_output_dataframe(rows)


def run():
    """Compute metrics."""
    # Parse argument
    parser = argparse.ArgumentParser()
    parser.add_argument("--production_data", type=str)
    parser.add_argument("--std_deviation_threshold", type=str)
    parser.add_argument("--signal_metrics", type=str)
    args = parser.parse_args()

    df = read_mltable_in_spark(args.production_data)

    signal_metrics = _compute_max_standard_deviation(
        df, float(args.std_deviation_threshold)
    )
    signal_metrics.show()

    save_spark_df_as_mltable(signal_metrics, args.signal_metrics)


if __name__ == "__main__":
    run()
