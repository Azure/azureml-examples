"""
onnxruntimetriton

Offers the class InferenceSession which can be used as a drop-in replacement for the ONNX Runtime
session object.

"""

import tritonclient.http as tritonhttpclient
import numpy as np


class NodeArg:
    def __init__(self, name, shape):
        self.name = name
        self.shape = shape


class InferenceSession:
    def __init__(self, path_or_bytes, sess_options=None, providers=[]):
        self.client = tritonhttpclient.InferenceServerClient("localhost:8000")
        model_metadata = self.client.get_model_metadata(model_name=path_or_bytes)

        self.request_count = 0
        self.model_name = path_or_bytes
        self.inputs = []
        self.outputs = []
        self.dtype_mapping = {}

        for (src, dest) in (
            (model_metadata["inputs"], self.inputs),
            (model_metadata["outputs"], self.outputs),
        ):
            for element in src:
                dest.append(NodeArg(element["name"], element["shape"]))
                self.dtype_mapping[element["name"]] = element["datatype"]

        self.triton_enabled = True

    def get_inputs(self):
        return self.inputs

    def get_outputs(self):
        return self.outputs

    def run(self, output_names, input_feed, run_options=None):
        inputs = []
        for key, val in input_feed.items():
            val = np.expand_dims(val, axis=0)
            input = tritonhttpclient.InferInput(key, val.shape, self.dtype_mapping[key])
            input.set_data_from_numpy(val)
            inputs.append(input)

        outputs = []

        for output_name in output_names:
            output = tritonhttpclient.InferRequestedOutput(output_name)
            outputs.append(output)

        res = self.client.async_infer(
            self.model_name, inputs, request_id=str(self.request_count), outputs=outputs
        )
        res = res.get_result()
        results = []
        for output_name in output_names:
            results.append(res.as_numpy(output_name))

        return results
