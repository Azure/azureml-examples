# Copyright (c) 2020-2021, NVIDIA CORPORATION. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
#  * Neither the name of NVIDIA CORPORATION nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
# EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
# PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
# OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import argparse
import sys
import os
import numpy as np
import tensorflow as tf
import gevent.ssl
import tritonclient.http as tritonhttpclient
import requests
import tokenization
from run_onnx_squad import *
import json

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_url")
    parser.add_argument("--token")
    parser.add_argument("--model", default="bert-squad-batch")
    args = parser.parse_args()

    scoring_uri = args.base_url[8:]
    triton_client = tritonhttpclient.InferenceServerClient(
        url=scoring_uri,
        ssl=True,
        ssl_context_factory=gevent.ssl._create_default_https_context,
    )

    headers = {}
    headers["Authorization"] = f"Bearer {args.token}"

    #print("token/key",args.token)
    print("scoring_uri",scoring_uri)

    # Check status of triton server
    health_ctx = triton_client.is_server_ready(headers=headers)
    print("Is server ready - {}".format(health_ctx))

    # Check status of model
    model_name = args.model
    status_ctx = triton_client.is_model_ready(model_name, "1", headers)
    print("Is model ready - {}".format(status_ctx))



    ## generate sample input data and save to json file 
    input_data = {
    "version": "1.4",
    "data": [
      {
        "paragraphs": [
          {
            "context": "In its early years, the new convention center failed to meet attendance and revenue expectations.[12] By 2002, many Silicon Valley businesses were choosing the much larger Moscone Center in San Francisco over the San Jose Convention Center due to the latter's limited space. A ballot measure to finance an expansion via a hotel tax failed to reach the required two-thirds majority to pass. In June 2005, Team San Jose built the South Hall, a $6.77 million, blue and white tent, adding 80,000 square feet (7,400 m2) of exhibit space",
            "qas": [
              {
                "question": "where is the businesses choosing to go?",
                "id": "1"
              },
              {
                "question": "how may votes did the ballot measure need?",
                "id": "2"
              },
              {
                "question": "By what year many Silicon Valley businesses were choosing the Moscone Center?",
                "id": "3"
              }
            ]
          }
        ],
        "title": "Conference Center"
      }
    ]
    }
    with open('inputs.json','w') as outfile:
         json.dump(input_data,outfile,indent=4)


    # preprocess inputs
    predict_file = 'inputs.json'
    # Use read_squad_examples method from run_onnx_squad to read the input file
    eval_examples = read_squad_examples(input_file=predict_file)
    
    max_seq_length = 256
    doc_stride = 128
    max_query_length = 64
    batch_size = 1
    n_best_size = 20
    max_answer_length = 30
    tf.gfile = tf.io.gfile  # this is used if using tf > 2.0
    
    vocab_file = os.path.join('endpoints/online/triton/batching/uncased_L-12_H-768_A-12', 'vocab.txt')
    tokenizer = tokenization.FullTokenizer(vocab_file=vocab_file, do_lower_case=True)


    # Use convert_examples_to_features method from run_onnx_squad to get parameters from the input 
    input_ids, input_mask, segment_ids, extra_data = convert_examples_to_features(eval_examples, tokenizer, 
                                                                              max_seq_length, doc_stride, max_query_length)

    # Get model metadata
    model_metadata = triton_client.get_model_metadata(model_name=model_name, headers=headers)
    input_meta = model_metadata["inputs"]
    output_meta= model_metadata["outputs"]

    n = len(input_ids)
    bs = batch_size
    all_results = []

    for idx in range(0,n):
        item = eval_examples[idx]
        input_mapping = {"unique_ids_raw_output___9:0": np.array([[item.qas_id]], dtype=np.int64),
            "input_ids:0": input_ids[idx:idx+bs],
            "input_mask:0": input_mask[idx:idx+bs],
            "segment_ids:0": segment_ids[idx:idx+bs]}

        inputs = []
        outputs = []

        #populate the inputs array
        for in_meta in input_meta:
            input_name = in_meta["name"]
            data = input_mapping[input_name]

            input = tritonhttpclient.InferInput(input_name,data.shape,in_meta["datatype"])
            input.set_data_from_numpy(data,binary_data=False)
            inputs.append(input)

        # populate the outputs array
        for out_meta in output_meta:
            output_name = out_meta["name"]
            output = tritonhttpclient.InferRequestedOutput(output_name,binary_data=False)
            outputs.append(output)


        result = triton_client.infer(model_name,inputs,outputs=outputs,headers=headers)

        pred0 = result.as_numpy("unstack:0")
        pred1 = result.as_numpy("unstack:1")
        in_batch = 1
        start_logits = [float(x) for x in pred0.flat]
        end_logits = [float(x) for x in pred1.flat]

        for i in range(0,in_batch):
            unique_id = len(all_results)
            all_results.append(RawResult(unique_id=unique_id,start_logits=start_logits,end_logits=end_logits))

    # postprocessing
    output_dir = 'predictions-triton'
    os.makedirs(output_dir, exist_ok=True)
    output_prediction_file = os.path.join(output_dir, "predictions_triton.json")
    output_nbest_file = os.path.join(output_dir, "nbest_predictions_triton.json")
    write_predictions(eval_examples, extra_data, all_results,
                      n_best_size, max_answer_length,
                      True, output_prediction_file, output_nbest_file)

    # print results
    with open(output_prediction_file) as json_file:  
        test_data = json.load(json_file)
        print(json.dumps(test_data, indent=2))

    
