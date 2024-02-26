import os
import json
import torch
import numpy as np

from transformers import AutoTokenizer, pipeline
import triton_python_backend_utils as pb_utils


class TritonPythonModel:
    def initialize(self, args):
        self.logger = pb_utils.Logger
        self.model_config = json.loads(args["model_config"])
        # self.model_params = self.model_config.get("parameters", {})
        
        default_max_gen_length = 200
        
        self.hf_model = "microsoft/phi-2"

        # Check for user-specified max length in model config parameters
        # self.max_output_length = int(
        #     self.model_params.get("max_output_length", {}).get(
        #         "string_value", default_max_gen_length
        #     )
        # )

        self.max_output_length = default_max_gen_length

        self.logger.log_info(f"Max response length (tokens): {self.max_output_length}")
        
        self.logger.log_info(f"Loading {self.hf_model}...")
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.hf_model,
                                                       trust_remote_code=True)

        self.pipeline = pipeline(
            "text-generation",
            model=self.hf_model,
            torch_dtype=torch.float32,
            tokenizer=self.tokenizer,
            device_map="auto",
            trust_remote_code=True
        )

    def execute(self, requests):
        responses = []
        for request in requests:

            input_tensor = pb_utils.get_input_tensor_by_name(request, "text_input")
            prompt = input_tensor.as_numpy()[0].decode("utf-8")

            response = self.generate(prompt)
            responses.append(response)

        return responses

    def generate(self, prompt):
        sequences = self.pipeline(
            prompt,
            do_sample=True,
            top_k=10,
            num_return_sequences=1,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.unk_token_id,
            max_length=self.max_output_length,
            truncation=True
        )
        output_tensors = []
        texts = []
        for i, seq in enumerate(sequences):
            text = seq["generated_text"]
            texts.append(text)
        tensor = pb_utils.Tensor("text_output", np.array(texts, dtype=np.object_))
        output_tensors.append(tensor)
        response = pb_utils.InferenceResponse(output_tensors=output_tensors)
        return response

    def finalize(self):
        print("Cleaning up...")

