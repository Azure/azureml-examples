#! /usr/bin/env python

import argparse
import gevent.ssl
import numpy as np
import requests
import io

import tritonclient.http as tritonhttpclient

if (__name__=='__main__'):
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_url')
    parser.add_argument('--token')
    parser.add_argument('--num_requests', type=int, default=1)
    args = parser.parse_args()

    headers = {"Content-Type": "application/octet-stream"}
    headers['Authorization'] = f'Bearer {args.token}'

    test_sample = requests.get("https://aka.ms/peacock-pic", allow_redirects=True).content

    test=np.array([test_sample], dtype=bytes)
    test = np.stack(test, axis=0)
    input = tritonhttpclient.InferInput('img_in_bytes', test.shape, 'BYTES')
    input.set_data_from_numpy(test)
    inputs = [input]
    headers = {"Authorization": f"Bearer  {args.token}"}
    if args.token:
        client = tritonhttpclient.InferenceServerClient(
            args.base_url,
            ssl=True,
            ssl_context_factory=gevent.ssl._create_default_https_context)
    else:
        client = tritonhttpclient.InferenceServerClient(args.base_url)
    outputs = [tritonhttpclient.InferRequestedOutput('label')]
    print(f'liveness check: {client.is_server_live(headers=headers)}')

    for i in range(args.num_requests):
        result = client.infer(model_name='ensemble', inputs=inputs, request_id='1', outputs=outputs, headers=headers)
        if i % 10 == 0:
            print(f"scoring check: {result.as_numpy('label')}, iteration {i}")
