import tritonhttpclient


def triton_init(url="localhost:8000"):
    """Initializes the triton client to point at the specified URL

    Parameter
    ----------
    url : str
        The URL on which to address the Triton server, defaults to
        localhost:8000
    """
    global triton_client
    triton_client = tritonhttpclient.InferenceServerClient(url)
    return triton_client


def get_model_info():
    """Gets metadata for all models hosted behind the Triton endpoint.
    Useful for confirming that your models were loaded into memory.

    Prints the data to STDOUT.
    """
    repo_index = triton_client.get_model_repository_index()
    for model in repo_index:
        model_name = model["name"]
        model_version = model["version"]
        (
            input_meta,
            input_config,
            output_meta,
            output_config,
        ) = parse_model_http(model_name=model_name, model_version=model_version)
        print(
            f"Found model: {model_name}, version: {model_version}, \
              input meta: {input_meta}, input config: {input_config}, \
              output_meta: {output_meta}, output config: {output_config}"
        )


def parse_model_http(model_name, model_version=""):
    """Check the configuration of a model to make sure it meets the
    requirements for an image classification network (as expected by
    this client)

    Arguments
    --------
    model_name : str
        Name of the model whose metadata you want to fetch

    model_version : str
        Optional, the version of the model, defaults to empty string.

    From https://github.com/triton-inference-server/server/blob/master/src/clients/python/examples/image_client.py  # noqa
    """
    model_metadata = triton_client.get_model_metadata(
        model_name=model_name, model_version=model_version
    )
    model_config = triton_client.get_model_config(
        model_name=model_name, model_version=model_version
    )

    return (
        model_metadata["inputs"],
        model_config["input"],
        model_metadata["outputs"],
        model_config["output"],
    )


def triton_infer(
    input_mapping,
    model_name,
    binary_data=False,
    binary_output=False,
    class_count=0,
):
    """Helper function for setting Triton inputs and executing a request

    Arguments
    ----------
    input_mapping : dict
        A dictionary mapping strings to numpy arrays. The keys should
        be the names of the model inputs, and the values should be the
        inputs themselves.

    model_name : str
        The name of the model on which you are running inference.

    binary_data : bool
        Whether you are expecting binary input and output. Defaults to False

    class_count : int
        If the model is a classification model, the number of output classes.
        Defaults to 0, indicating this is not a classification model.

    Returns
    ----------
    res : InferResult
        Triton inference result containing output from running prediction
    """
    input_meta, _, output_meta, _ = parse_model_http(model_name)

    inputs = []
    outputs = []

    # Populate the inputs array
    for in_meta in input_meta:
        input_name = in_meta["name"]
        data = input_mapping[input_name]

        input = tritonhttpclient.InferInput(input_name, data.shape, in_meta["datatype"])

        input.set_data_from_numpy(data, binary_data=binary_data)
        inputs.append(input)

    # Populate the outputs array
    for out_meta in output_meta:
        output_name = out_meta["name"]
        output = tritonhttpclient.InferRequestedOutput(
            output_name, binary_data=binary_output, class_count=class_count
        )
        outputs.append(output)

    # Run inference
    res = triton_client.infer(model_name, inputs, request_id="0", outputs=outputs)

    return res
