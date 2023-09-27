# Deploy an integrated inference container 
This example shows how to deploy a fully-integrated inference BYOC image using TFServing with the model integrated into the container itself and both `model` and `code` absent from the deployment yaml.

## How to deploy
To deploy this example execute the deploy-custom-container-half-plus-two-integrated.sh script in the CLI directory. 

## Testing
The endpoint can be tested using the code at the end of the deployment script:
```bash
curl -d @sample-data.json -H "Content-Type: application/json" -H "Authorization: Bearer $KEY" $SCORING_URL
```

The inputs are a list of tensors of dimension 2. 
```json
{
    "inputs" : [
                    [[1,2,3,4]],
                    [[0,1,1,1]]
               ]
}
``` 

## Model 
This model is a simple [Half Plus Two](https://www.tensorflow.org/tfx/serving/docker) model which returns `2X+0.5` for input tensors of dimension 2. 

```python 
class HPT(tf.Module):
    @tf.function(input_signature=[tf.TensorSpec(shape=[None,None], dtype=tf.float32)])
    def __call__(self, x):
        return tf.math.add(tf.math.multiply(x, 0.5),2)
```

## Image
This is a BYOC TFServing image with the model integrated into the container itself. 