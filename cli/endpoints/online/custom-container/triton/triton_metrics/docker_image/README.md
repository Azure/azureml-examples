# Docker Image

This Docker Image consists of 3 components:
1. A Nvidia Triton Server collecting and exposing metrics.
2. A Prometheus client that receives those metrics and sends it to a Python script.
3. A Python script that collects those metrics and sends them to Application Insights.

# How to Run Build File

build.sh {IMAGE_NAME} {IMAGE_VERSION}

# Quick Start

docker run --rm -p8000:8000 -p8001:8001 -p8002:8002 -p 9090:9090 --env '{INSTRUMENTATION_KEY}' {IMAGE_NAME}

# Architechture
Request Flow
------------
![Flow Chart](flow_visualization.png)

Diagram of Architecture
-----------------------
![Model infrencing data](architecture_drawing.png)

Explination of components
-------------------------

The image is split into 3 main parts: the customer's Triton server, the Prometheus client, and the Python script. 

Triton Server: 

The Triton software enables users to run a variety of machine learning models and deploy them. It also collects metrics from the running models as well as the CPU and GPU performance and exposes them on port 8002 on the local host. 

Prometheus: 

 Prometheus scrapes the Triton metrics at a specified time interval and exposes them in a JSON format that is easily read by the Python script. It also enables us to define custom rules on how we want the metrics to be aggregated. The reason for the Prometheus client is that the Triton server provides its metrics in a Prometheus format. So, in order to consume those metrics, we had to use the Prometheus API through the Prometheus client. 

Python Script: 

The Python script receives the metrics from Prometheus via a GET request, then sends those metrics to the Application Insights endpoint with a POST request. The reason for the addition of the Python script is that the Prometheus client cannot push the metric data directly to Application Insights but can be collected and sent using Python. 

Other Components' Architechtures
--------------------------------

* `Triton Architechture`: https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/user_guide/architecture.html
* `Prometheus Architechture`: https://prometheus.io/docs/introduction/overview/

# Triton Metrics Feature
This version of the Triton Docker image sends metric data from Triton Server to Application Insights in the Azure portal. There is a metrics.json file in the image that contains 3 of the default metrics provided by Triton. The 3 metrics in the image are an example, and the user can choose which metrics they want to be pushed to App Insights by providing their own metrics.json file. 

The format of it is as follows: 
{
    "customMetrics": [
    "nv_inference_request_success", 
    "nv_inference_request_duration_us", 
    "nv_cpu_utilization"
                     ]
}

# List of Available Triton Metrics
By default every metric besides CPU and GPU utilization are an accumulation over time. As a part of my feature I also added a version of the metrics that takes the instant rate of the metric every 60 seconds; these are shown by the "irate" versions of the default metrics.

Category: Count
- nv_inference_request_success
- nv_inference_request_failure
- nv_inference_count
- nv_inference_exec_count

- nv_inference_request_success_irate
- nv_inference_request_failure_irate
- nv_inference_count_irate
- nv_inference_exec_count_irate


Category: Latency
- nv_inference_request_duration_us
- nv_inference_queue_duration_us
- nv_inference_compute_input_duration_us
- nv_inference_compute_infer_duration_us
- nv_inference_compute_output_duration_us

- nv_inference_request_duration_us_irate
- nv_inference_queue_duration_us_irate
- nv_inference_compute_input_duration_us_irate
- nv_inference_compute_infer_duration_us_irate
- nv_inference_compute_output_duration_us_irate


Category: Summary-latency
- nv_inference_request_summary_us
- nv_inference_queue_summary_us
- nv_inference_compute_input_summary_us
- nv_inference_compute_infer_summary_us
- nv_inference_compute_output_summary_us

- nv_inference_request_summary_us_irate
- nv_inference_queue_summary_us_irate
- nv_inference_compute_input_summary_us_irate
- nv_inference_compute_infer_summary_us_irate
- nv_inference_compute_output_summary_us_irate


Category: GPU metrics
- nv_gpu_power_usage
- nv_gpu_power_limit
- nv_energy_consumption
- nv_gpu_utilization
- nv_gpu_memory_total_bytes
- nv_gpu_memory_used_bytes

Category: CPU metrics
- nv_cpu_utilization
- nv_cpu_memory_total_bytes
- nv_cpu_memory_used_bytes