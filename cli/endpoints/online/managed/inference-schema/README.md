# Inference Schema
In this example, we use the [Inference Schema](https://github.com/Azure/InferenceSchema) package to facilitate automatic Swagger generation and parameter casting for Managed Online Endpoints.

To run this example end-to-end, execute the [deploy-inference-schema](../../../../deploy-moe-inference-schema.sh) script. 

[Inference Schema](https://github.com/Azure/InferenceSchema) is an open source library for ML applications maintained by Travis Angevine [@trangevi](https://github.com/trangevi) that streamlines schema design and development for ML applications and offers features such as parameter type definition, automatic type conversion, and schema/swagger file generation. Using Inference Schema, users can easily define parameter types and associate them to functions with input and output decorators. 

Inference Schema integrates directly with AzureML endpoints. User `run` functions with Inference Schema decorators can be defined with an arbitrary number of arguments and receive automatic swagger file generation at `/swagger.json`. 

# General Usage 

## Function Decorators
The decorators `input_schema` and `output_schema` are used to define the schema. The `input_schema` decorator can be stacked multiple times as in [score-standard](code/score-standard.py) to correspond to multiple function arguments in the run function. 

## Parameter Types
There are 4 core parameter types available:
- StandardPythonParameterType
- PandasParameterType
- NumpyParameterType
- SparkParameterType

It is possible to nest parameter types by wrapping them in a list or dict and a `StandardParameterType` - see [Standard/Multiple Parameters](#example-standardmultiple-parameters). 


## Swagger
The automatically-generated Swagger can be retrieved by default at `/swagger.json` with an optional `version` HTTP parameter.

More details about Swagger can be found in the [deploy-moe-swagger](../../../../deploy-moe-swagger.sh) example. 


# Example: Numpy

## Parameters
The [score-numpy.py](code/score-numpy.py) script declares a single input parameter `iris` as a `NumpyParameterType` and a nested `StandardParameterType` as output. 
```python
@input_schema(
    param_name="iris",
    param_type=NumpyParameterType(np.array([[7.2, 3.2, 6.0, 1.8]]))
)
@output_schema(
    output_type=StandardPythonParameterType({
        "Category" : ["Virginica"]
    })
)
```

## Request
When the following [sample input](sample-inputs/numpy.json) is sent, it is automatically validated and converted to a Numpy array. Validation can be toggled using the `enforce_column_type` and `enforce_shape` arguments to `NumpyParameterType`. 

```json
{"iris": [[7.2, 3.2, 6.0, 1.8], [4.2, 3.5, 1.0, 3.0]]}
```

# Example: Standard/Multiple Parameters
In [score-standard.py](code/score-standard.py), the run function accepts multiple arguments by applying the `input_schema` decorator several times. 

## Parameters

```python 
@input_schema(
    param_name="sepal_length",
    param_type=StandardPythonParameterType([7.2])
)
@input_schema(
    param_name="sepal_width",
    param_type=StandardPythonParameterType([3.2])
)
@input_schema(
    param_name="petal_length",
    param_type=StandardPythonParameterType([6.0])
)
@input_schema(
    param_name="petal_width",
    param_type=StandardPythonParameterType([1.8])
)
@output_schema(
    output_type=StandardPythonParameterType({
        "Category" : ["Virginica"]
    })
)
def run(sepal_length, sepal_width, petal_length, petal_width):
```
## Request 
```json
{
    "sepal_length": [7.2, 4.2],
    "sepal_width": [3.2,3.5], 
    "petal_length": [6.0,1.0],
    "petal_width": [1.8,3.0]
}
```

# Example: Pandas

In [score-pandas.py](code/score-standard.py), the run function is designed to take a Pandas dataframe as a single argument input. 

## Parameters

```python
@input_schema(
    param_name="iris",
    param_type=PandasParameterType(pd.DataFrame({
        "sepal_length": [7.2],
        "sepal_width": [3.2],
        "petal_length": [6.0],
        "petal_width": [1.8]})
    )
)
@output_schema(
    output_type=StandardPythonParameterType({
        "Category" : ["Virginica"]
    })
)
``` 

## Request 

When the following sample input is passed, it is automatically validated and converted to a DataFrame.

```json 
{
    "iris": {
        "sepal_length": {
            "0": 7.2,
            "1": 4.2
        },
        "sepal_width": {
            "0": 3.2,
            "1": 3.5
        },
        "petal_length": {
            "0": 6.0,
            "1": 1.0
        },
        "petal_width": {
            "0": 1.8,
            "1": 3.0
        }
    }
}
```

