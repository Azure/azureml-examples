# Object Detection Using Built-In Component

## Overview

Using built-in components, you can train an object detection model (and other vision models) without writing any code. You can refer to documentation [here](built-in-vision-components.md) for more information on built-in vision components.


## How to Run

1. Prepare the data

    You first need to prepare the data: The image based datasets need to be uploaded and later the JSONL dataset needs to be generated based on the uploaded images URIs.

    You do this by running the prepare_data.py available in this folder, with a command like:

    ``` bash
    python prepare_data.py --workspace-name [YOUR_AZURE_WORKSPACE] --resource-group [YOUR_AZURE_RESOURCE_GROUP] --subscription [YOUR_AZURE_SUBSCRIPTION]
    ```

1. Run the CLI command

    Run the CLI command pointing to the pipeline .YML file in this folder plus the Azure ML IDs needed.

    Note: Your compute cluster should be GPU-based when training with images or text. You need to specify/change the name of the cluster in the .YML file (compute: azureml:gpu-cluster).

    ``` bash
    az ml job create --file ./object-detection-using-built-in-component.yml --workspace-name [YOUR_AZURE_WORKSPACE] --resource-group [YOUR_AZURE_RESOURCE_GROUP] --subscription [YOUR_AZURE_SUBSCRIPTION]
    ```
