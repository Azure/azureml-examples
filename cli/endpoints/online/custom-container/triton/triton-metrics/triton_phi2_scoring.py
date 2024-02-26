import argparse
import gevent.ssl
import numpy as np
import tritonclient.http as tritonhttpclient
from tritonclient.utils import np_to_triton_dtype

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_url")
    parser.add_argument("--token")
    parser.add_argument("--prompt", type=str, default="An apple a day")
    args = parser.parse_args()

    scoring_uri = args.base_url[8:]
    triton_client = tritonhttpclient.InferenceServerClient(
        url=scoring_uri,
        ssl=True,
        ssl_context_factory=gevent.ssl._create_default_https_context,
    )

    headers = {}
    headers["Authorization"] = f"Bearer {args.token}"

    # Check status of triton server
    health_ctx = triton_client.is_server_ready(headers=headers)
    print("Is server ready - {}".format(health_ctx))

    # Check status of model
    model_name = "phi-2"
    status_ctx = triton_client.is_model_ready(model_name, "1", headers)
    print("Is model ready - {}".format(status_ctx))

    prompts = [args.prompt]
    text_obj = np.array(prompts, dtype="object")

    # Populate inputs and outputs
    input = tritonhttpclient.InferInput("text_input", text_obj.shape, np_to_triton_dtype(text_obj.dtype))
    input.set_data_from_numpy(text_obj)
    inputs = [input]
    output = tritonhttpclient.InferRequestedOutput("text_output")
    outputs = [output]

    result = triton_client.infer(model_name, inputs, outputs=outputs, headers=headers)
    response = str(result.as_numpy("text_output"))
    print(response)
