# Use OpenAPI with Managed Online Endpoints

To run this example end-to-end, execute the [deploy-openapi.sh](../../../../deploy-moe-openapi.sh) script. The code snippets below are taken from this script, and you must change directory to the [`CLI`]('../../../..') directory for the code to run. 
 
## Overview

This example demonstrates how to work with with OpenAPI and Managed Online Endpoints using both automatically-generated and custom Swagger files. 

The AzureML Inference Server automatically generates swagger files for scoring scripts that use [Inference Schema](https://github.com/Azure/InferenceSchema). In this example, a simple Inference Schema-decorated [scoring script](openapi/decorated/code/score.py) is used. For more complex examples, refer to the [Inference Schema example](../inference-schema). 

Managed Online Endpoints can also return user-defined swagger files.  

## Prerequisites

- An Azure account with an active subscription. [Create an account for free](https://azure.microsoft.com/free/?WT.mc_id=A261C142F)
- An Azure ML workspace. [Check this notebook for creating a workspace](/sdk/resources/workspace/workspace.ipynb)
- Install and configure the Azure CLI and ML (v2) extension. For more information, see [Install, set up, and use the 2.0 CLI](https://learn.microsoft.com/en-us/azure/machine-learning/how-to-configure-cli?tabs=public)


## Get started

### Set variables

```bash
RAND=`echo $RANDOM`
ENDPOINT_NAME="endpt-moe-$RAND"
``` 

### Create an endpoint
```bash
az ml online-endpoint create -n $ENDPOINT_NAME
```

### Get the key, scoring, and OpenAPI urls
The default OpenAPI URL is `/swagger.json`. 

```bash
echo "Getting access key..."
KEY=$(az ml online-endpoint get-credentials -n $ENDPOINT_NAME --query primaryKey -o tsv )

echo "Getting scoring url..."
SCORING_URL=$(az ml online-endpoint show -n $ENDPOINT_NAME --query scoring_uri -o tsv )
echo "Scoring url is $SCORING_URL"

echo "Getting OpenAPI (Swagger) url..."
SWAGGER_URL=$(az ml online-endpoint show -n $ENDPOINT_NAME --query openapi_uri -o tsv )
echo "OpenAPI (Swagger) url is $OPENAPI_URL"
```

## Auto-Generated Swagger
The AzureML Inference Server automatically generates swagger files for scoring scripts that use [Inference Schema](https://github.com/Azure/InferenceSchema). In this example, a simple Inference Schema-decorated [scoring script](decorated/code/score.py) is used. For more complex examples, refer to the [Inference Schema example](../inference-schema).  

In this deployment, the [`code-decorated`](code-decorated) folder contains only a `score.py` file without a user-supplied swagger file. The run function of this `score.py` file is decorated with Inference Schema decorators: 

```python
@input_schema(
    param_name="data",
    param_type=NumpyParameterType(np.array([[1,2,3,4,5,6,7,8,9,10]]))
)
@output_schema(
    param_type=StandardPythonParameterType(
        {"output": [1.0, 1.0]}
    )
)
def run(data):
    logging.info("model 1: request received")
    result = model.predict(data)
    logging.info("Request processed")
    return {"output": result.tolist()}
```

### Create the deployment
The deployment template is [deployment.yml](deployment.yml): 

```yaml 
$schema: https://azuremlschemas.azureedge.net/latest/managedOnlineDeployment.schema.json
name: openapi
endpoint_name: <my-endpoint-name>
model:
  path: ../../model-1/model
code_configuration:
  code: <CODE_DIR>
  scoring_script: score.py
environment:
  image: mcr.microsoft.com/azureml/minimal-ubuntu22.04-py39-cpu-inference
  conda_file: env.yml
instance_type: Standard_DS3_v2
instance_count: 1
```

Note that the code directory is set to to `code-decorated`. 

```bash
az ml online-deployment create -f endpoints/online/managed/openapi/deployment.yml \
  --set endpoint_name=$ENDPOINT_NAME \
  --set code_configuration.code=code-decorated \
  --all-traffic
``` 

### Test the deployment
```bash
curl -H "Authorization: Bearer $KEY" -H "Content-Type: application/json" \
    -d @endpoints/online/model-1/sample-request.json $SCORING_URL
```

### Consume Swagger

Swagger files are made available by default at the API endpoint `/swagger.json`.

The swagger file is accessed through a GET request: 
```bash
curl -H "Authorization: Bearer $KEY"  $OPENAPI_URL
``` 

The specific route for an endpoint can be retrieved from the `openapi_uri` property of the endpoint with the following command: 

```bash
az ml online-endpoint show -n $ENDPOINT_NAME --query openapi_uri -o tsv 
``` 

Specific OpenAPI versions can be retrieved by adding a `version` parameter like so: 

```bash
curl -H "Authorization: Bearer $KEY" "$OPENAPI_URL?version=3"
```

## Custom Swagger
[Custom swagger files](code-custom/swagger2.json) can be integrated by including them at the root of the `code` directory. The custom file should be named swagger<version>.json. 

In this deployemnt the [code-custom](code-custom) directory contains both a `score.py` file (without Inference Schema decorators) as well as a custom swagger file called `swagger2.json`: 

```json
{
    "swagger": "2.0",
    "info": {
        "title": "ML service",
        "description": "A custom API description",
        "version": "1.0"
    },
    "schemes": [
        "https"
    ],
...
```

Update the deployment with the `code-custom` code directory: 

```bash
az ml online-deployment update -f endpoints/online/managed/openapi/deployment.yml \
  --set endpoint_name=$ENDPOINT_NAME \
  --set code_configuration.code=code-custom
``` 

The custom swagger file we provided is supplied at `/swagger.json` even though the file is named `swagger2.json`. The requested version is controlled by the version parameter of the request, see section 3.7. 

```bash
curl -H "Authorization: Bearer $KEY" $SWAGGER_URL
``` 