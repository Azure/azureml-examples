# Use OpenAPI with Managed Online Endpoints

This example demonstrates how to work with with OpenAPI and Managed Online Endpoints using both automatically-generated and custom Swagger files. 

To run this example end-to-end, execute the [deploy-openapi.sh](../../../../deploy-moe-openapi.sh) script. 

# Auto-generation
The AzureML Inference Server automatically generates swagger files for scoring scripts that use [Inference Schema](https://github.com/Azure/InferenceSchema). In this example, a simple Inference Schema-decorated [scoring script](decorated/code/score.py) is used. For more complex examples, refer to the [Inference Schema example](../inference-schema).  

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

# Consuming
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

# Custom
[Custom swagger files](custom/code) can be integrated by including them at the root of the `code` directory. The custom file should be named swagger<version>.json. 