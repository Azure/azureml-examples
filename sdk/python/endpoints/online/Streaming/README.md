# Streaming Responses from Managed Online Endpoints

Using generators and `AMLResponse` objects, Managed Online Endpoints can support streaming the results of an inference request incrementally.

## Background

In basic scoring scripts, the `run` function returns the entire result of an inference operation at once:

```python
def run(raw_data):
    data = json.loads(raw_data)["data"]
    data = numpy.array(data)
    result = model.predict(data)
    logging.info("Request processed")
    return result.tolist()
```

To stream inference results incrementally, a generator can be passed to an `AMLResponse` object and returned by the run function instead:

```python

@rawhttp
def run(req: AMLRequest):
    print(req)
    return AMLResponse(generate(), 200)
```

# Default (Buffered) Response

By default

# Unbuffered Response

