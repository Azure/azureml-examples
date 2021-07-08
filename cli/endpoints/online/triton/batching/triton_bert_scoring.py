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
    model_name = "bert-si-onnx"
    status_ctx = triton_client.is_model_ready(model_name, "1", headers)
    print("Is model ready - {}".format(status_ctx))

    # preprocess inputs
    predict_file = "endpoints/online/triton/single-model/inputs.json"
    # Use read_squad_examples method from run_onnx_squad to read the input file
    eval_examples = read_squad_examples(input_file=predict_file)

    max_seq_length = 256
    doc_stride = 128
    max_query_length = 64
    batch_size = 1
    n_best_size = 20
    max_answer_length = 30
    tf.gfile = tf.io.gfile  # this is used if using tf > 2.0

    vocab_file = os.path.join(
        "endpoints/online/triton/single-model/uncased_L-12_H-768_A-12", "vocab.txt"
    )
    tokenizer = tokenization.FullTokenizer(vocab_file=vocab_file, do_lower_case=True)

    # Use convert_examples_to_features method from run_onnx_squad to get parameters from the input
    input_ids, input_mask, segment_ids, extra_data = convert_examples_to_features(
        eval_examples, tokenizer, max_seq_length, doc_stride, max_query_length
    )

    # Get model metadata
    model_metadata = triton_client.get_model_metadata(
        model_name=model_name, headers=headers
    )
    input_meta = model_metadata["inputs"]
    output_meta = model_metadata["outputs"]

    n = len(input_ids)
    bs = batch_size
    all_results = []

    for idx in range(0, n):
        item = eval_examples[idx]
        input_mapping = {
            "unique_ids_raw_output___9:0": np.array([[item.qas_id]], dtype=np.int64),
            "input_ids:0": input_ids[idx : idx + bs],
            "input_mask:0": input_mask[idx : idx + bs],
            "segment_ids:0": segment_ids[idx : idx + bs],
        }

        inputs = []
        outputs = []

        # populate the inputs array
        for in_meta in input_meta:
            input_name = in_meta["name"]
            data = input_mapping[input_name]

            input = tritonhttpclient.InferInput(
                input_name, data.shape, in_meta["datatype"]
            )
            input.set_data_from_numpy(data, binary_data=False)
            inputs.append(input)

        # populate the outputs array
        for out_meta in output_meta:
            output_name = out_meta["name"]
            output = tritonhttpclient.InferRequestedOutput(
                output_name, binary_data=False
            )
            outputs.append(output)

        result = triton_client.infer(
            model_name, inputs, outputs=outputs, headers=headers
        )

        pred0 = result.as_numpy("unstack:0")
        pred1 = result.as_numpy("unstack:1")
        in_batch = 1
        start_logits = [float(x) for x in pred0.flat]
        end_logits = [float(x) for x in pred1.flat]

        for i in range(0, in_batch):
            unique_id = len(all_results)
            all_results.append(
                RawResult(
                    unique_id=unique_id,
                    start_logits=start_logits,
                    end_logits=end_logits,
                )
            )

    # postprocessing
    output_dir = "predictions-triton"
    os.makedirs(output_dir, exist_ok=True)
    output_prediction_file = os.path.join(output_dir, "predictions_triton.json")
    output_nbest_file = os.path.join(output_dir, "nbest_predictions_triton.json")
    write_predictions(
        eval_examples,
        extra_data,
        all_results,
        n_best_size,
        max_answer_length,
        True,
        output_prediction_file,
        output_nbest_file,
    )

    # print results
    with open(output_prediction_file) as json_file:
        test_data = json.load(json_file)
        print(json.dumps(test_data, indent=2))
