#!/usr/bin/env python3

import requests
import json
import os
import time
import logging
from datetime import datetime, timezone, timedelta

ENV_AML_APP_INSIGHTS_KEY = "AML_APP_INSIGHTS_KEY"

PROMETHEUS_URL = "http://localhost:9090"
APP_INSIGHTS_URL = "https://dc.services.visualstudio.com/v2/track"
VALID_METRICS_FILEPATH = "/var/azureml-util/metrics_utilities/valid_metrics.json"

logging.basicConfig(
    format="%(asctime)s %(levelname)-8s %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
)

# Link to available metrics: https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/user_guide/metrics.html
# Metric names are case sensitive
def get_metric_labels():
    metric_labels = []
    try:
        # Checks if the file with valid metrics exists
        if os.path.exists(VALID_METRICS_FILEPATH):
            with open(VALID_METRICS_FILEPATH) as json_file:
                metric_labels = json.load(json_file)

        if not metric_labels:
            # Query all available metrics from prometheus
            response = requests.get(PROMETHEUS_URL + "/api/v1/label/__name__/values")
            if response.ok:
                json_response = json.loads(response.text)
                metric_labels = json_response["data"]
                logging.info("Available Triton metrics to track")
                logging.info(metric_labels)
                # Save all valid metrics to a file
                with open(VALID_METRICS_FILEPATH, "w") as json_file:
                    json.dump(metric_labels, json_file)
            else:
                logging.debug(response.text)

        return metric_labels
    except Exception as exception:
        logging.debug(exception)


def query_prometheus(metric_name):
    try:
        response = requests.get(
            PROMETHEUS_URL + "/api/v1/query", params={"query": metric_name}
        )

        if not response.ok:
            logging.debug(response.text)
            return []

        metric = json.loads(response.text)

        # Check if metric data exists
        if len(metric["data"]["result"]) == 0:
            logging.info(metric_name + " is not part of available Triton metrics")
            return []

        # Some of the metrics are returned by model
        baseDataList = []
        for metric_per_model in metric["data"]["result"]:
            metric_value = metric_per_model["value"][1]

            baseData = {
                "metrics": [
                    {
                        "name": metric_name,
                        "value": float(metric_value),
                    }
                ]
            }

            # Check if model specific metric information is returned
            if "model" in metric_per_model["metric"]:
                baseData["properties"] = {
                    "model": metric_per_model["metric"]["model"],
                    "version": metric_per_model["metric"]["version"],
                }

            baseDataList.append(baseData)

        return baseDataList
    except Exception as exception:
        logging.debug(exception)


def push_to_appInsights(json_body):
    try:
        postRequest = requests.post(APP_INSIGHTS_URL, json=json_body)

        if not postRequest.ok:
            logging.debug(postRequest.text)

    except Exception as exception:
        logging.debug(exception)


app_insights_key = os.environ.get(ENV_AML_APP_INSIGHTS_KEY)

# Wait for other scripts to start
max_time = datetime.now() + timedelta(seconds=180)
while datetime.now() < max_time:
    time.sleep(0.5)
    metric_labels = get_metric_labels()
    if metric_labels:
        break

for metric_label in metric_labels:
    metric_data_list = query_prometheus(metric_label)
    for metric_data in metric_data_list:
        postBody = {
            "iKey": app_insights_key,
            "name": "Microsoft.ApplicationInsights.Event",
            "time": datetime.now(timezone.utc).isoformat(),
            "data": {"baseType": "MetricData", "baseData": metric_data},
        }

        push_to_appInsights(postBody)

logging.info("Successfully pushed metrics to app insights.")
