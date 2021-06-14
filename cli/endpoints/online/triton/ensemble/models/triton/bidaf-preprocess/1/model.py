import nltk
import json
import numpy as np

from nltk import word_tokenize

import triton_python_backend_utils as pb_utils


class TritonPythonModel:
    """Your Python model must use the same class name. Every Python model
    that is created must have "TritonPythonModel" as the class name.
    """

    def initialize(self, args):
        """`initialize` is called only once when the model is being loaded.
        Implementing `initialize` function is optional. This function allows
        the model to intialize any state associated with this model.

        Parameters
        ----------
        args : dict
          Both keys and values are strings. The dictionary keys and values are:
          * model_config: A JSON string containing the model configuration
          * model_instance_kind: A string containing model instance kind
          * model_instance_device_id: A string containing model instance device ID
          * model_repository: Model repository path
          * model_version: Model version
          * model_name: Model name
        """

        # You must parse model_config. JSON string is not parsed here
        self.model_config = model_config = json.loads(args["model_config"])

        # Get OUTPUT0 configuration
        output0_config = pb_utils.get_output_config_by_name(model_config, "OUTPUT0")

        # Get OUTPUT1 configuration
        output1_config = pb_utils.get_output_config_by_name(model_config, "OUTPUT1")

        # Get OUTPUT2 configuration
        output2_config = pb_utils.get_output_config_by_name(model_config, "OUTPUT2")

        # Get OUTPUT3 configuration
        output3_config = pb_utils.get_output_config_by_name(model_config, "OUTPUT3")

        # Convert Triton types to numpy types
        self.output0_dtype = pb_utils.triton_string_to_numpy(
            output0_config["data_type"]
        )
        self.output1_dtype = pb_utils.triton_string_to_numpy(
            output1_config["data_type"]
        )
        self.output2_dtype = pb_utils.triton_string_to_numpy(
            output2_config["data_type"]
        )
        self.output3_dtype = pb_utils.triton_string_to_numpy(
            output3_config["data_type"]
        )

        # Get model repository path to read labels
        self.model_repository = model_repository = args["model_repository"]
        print(model_repository)

        # Initialize tokenizer
        nltk.download("punkt")

    def tokenize(self, text):

        tokens = word_tokenize(text)

        # split into lower-case word tokens, in numpy array with shape of (seq, 1)
        words = np.array([w.lower() for w in tokens], dtype=np.object_).reshape(-1, 1)

        # split words into chars, in numpy array with shape of (seq, 1, 1, 16)
        chars = [[c for c in t][:16] for t in tokens]
        chars = [cs + [""] * (16 - len(cs)) for cs in chars]
        chars = np.array(chars, dtype=np.object_).reshape(-1, 1, 1, 16)

        return words, chars

    def execute(self, requests):
        """
        Parameters
        ----------
        requests : list
          A list of pb_utils.InferenceRequest

        Returns
        -------
        list
          A list of pb_utils.InferenceResponse. The length of this list must
          be the same as `requests`
        """

        output0_dtype = self.output0_dtype
        output1_dtype = self.output1_dtype
        output2_dtype = self.output2_dtype
        output3_dtype = self.output3_dtype

        responses = []

        # Every Python backend must iterate over everyone of the requests
        # and create a pb_utils.InferenceResponse for each of them.
        for request in requests:
            # Get INPUT0
            in_0 = pb_utils.get_input_tensor_by_name(request, "INPUT0")
            context = in_0.as_numpy().astype(str)
            print(context)

            # Get INPUT1
            in_0 = pb_utils.get_input_tensor_by_name(request, "INPUT1")
            query = in_0.as_numpy().astype(str)
            print(query)

            cw, cc = self.tokenize(context[0])
            qw, qc = self.tokenize(query[0])

            out_0 = np.array(qw, dtype=output0_dtype)
            out_1 = np.array(cc, dtype=output1_dtype)
            out_2 = np.array(qc, dtype=output2_dtype)
            out_3 = np.array(cw, dtype=output3_dtype)

            # Create output tensors. You need pb_utils.Tensor objects to create pb_utils.InferenceResponse.
            out_tensor_0 = pb_utils.Tensor("OUTPUT0", out_0)
            out_tensor_1 = pb_utils.Tensor("OUTPUT1", out_1)
            out_tensor_2 = pb_utils.Tensor("OUTPUT2", out_2)
            out_tensor_3 = pb_utils.Tensor("OUTPUT3", out_3)

            inference_response = pb_utils.InferenceResponse(
                output_tensors=[out_tensor_0, out_tensor_1, out_tensor_2, out_tensor_3]
            )
            responses.append(inference_response)

        return responses

    def finalize(self):
        """`finalize` is called only once when the model is being unloaded.
        Implementing `finalize` function is OPTIONAL. This function allows
        the model to perform any necessary clean ups before exit.
        """
        print("Cleaning up...")
