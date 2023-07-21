# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------

import json
import os
import psutil
import requests
import sys
import time

import logging
from pathlib import Path
from subprocess import PIPE, STDOUT
from subprocess import run as subprocess_run
from typing import Tuple
from text_generation import Client


# Configure logger
logger = logging.getLogger(__name__)
format_str = "%(asctime)s [%(module)s] %(funcName)s %(lineno)s: %(levelname)-8s [%(process)d] %(message)s"
formatter = logging.Formatter(format_str)
stream_handler = logging.StreamHandler()
stream_handler.setFormatter(formatter)
logger.setLevel(logging.DEBUG)
logger.addHandler(stream_handler)


PARENT_DIR = Path().parent.resolve()
# MODEL_LAUNCH_DETAILS = "model_launch_details.json"

PORT = 80
LOCAL_HOST_URI = f"http://0.0.0.0:{PORT}"
MODEL_ID = "model_id"
NUM_SHARD = "num_shard"
CLIENT_TIMEOUT = "timeout"
MAX_REQUEST_TIMEOUT = 90  # 90s

client = None

# print all env variables
for k, v in os.environ.items():
    print(f"{k}={v}")


def run_command(cmd: str) -> Tuple[int, str]:
    """Run the command and returns the result."""
    logger.info(f"run_command: executing {cmd}")
    result = subprocess_run(
        cmd,
        shell=True,
        stdout=PIPE,
        stderr=STDOUT,
        encoding=sys.stdout.encoding,
        errors="ignore",
    )
    return result


def get_init_args():
    # model_info_path = PARENT_DIR / MODEL_LAUNCH_DETAILS
    # with open(model_info_path) as f:
    #     model_info = json.load(f)

    return {
        MODEL_ID: "mlflow_model_folder/data/model",
        CLIENT_TIMEOUT: MAX_REQUEST_TIMEOUT
    }


def is_server_healthy():
    # use psutil to go trhough active process 

    process_name = "text-generation-launcher"
    WAIT_TIME = 20
    RETRY_COUNT = 5
    count = 0
    while count < RETRY_COUNT and process_name not in [p.name() for p in psutil.process_iter()]:
        logger.warning(f"Process {process_name} is not running. Sleeping for {WAIT_TIME}s and retrying")
        time.sleep(WAIT_TIME)
        count += 1
    if count >= RETRY_COUNT:
        total_dur = RETRY_COUNT * WAIT_TIME
        raise Exception(f"Sever process not running after waiting for {total_dur}. Terminating")

    logger.info(f"Server process {process_name} running. Hitting endpoint with 5s delay")
    time.sleep(5)

    payload_dict = {
        "inputs": "Meaning of life is",
        "parameters":{"max_new_tokens":2}
    }

    json_str = json.dumps(payload_dict)

    try:
        response = requests.post(
            url=LOCAL_HOST_URI,
            data=json_str,
            headers={
                "Content-Type": "application/json"
            }
        )
        logger.info(f"response status code: {response.status_code}")
        if response.status_code == 200 or response.status_code == 201:
            return True
    except Exception as e:
        logger.warning(f"Test request failed. Error {e}")
    return False


def init():
    global client

    try:
        model_id = configs.get(MODEL_ID)
        # num_shards = configs.get(NUM_SHARD)
        client_timeout = configs.get(CLIENT_TIMEOUT)

        assert model_id is not None or "model_id is not provided for scoring"

        model_path = os.path.join(os.getenv("AZUREML_MODEL_DIR", ""), model_id)
        logger.info(f"Loading model from path {model_path}")
        logger.info(f"List model_path = {os.listdir(model_path)}")

        cmd = f"text-generation-launcher --model-id {model_path}"
        # if num_shards:
        #     cmd += f" --num-shard {num_shards}"
        cmd += " &"

        logger.info("Starting server")
        os.system(cmd)
        time.sleep(20)

        # TODO: Make sure server is up and running
        # Add heart beat for the server
        # Once we get successful till timeout is reached

        WAIT_TIME = 60
        while not is_server_healthy():
            logger.info(f"Server not up. Waiting for {WAIT_TIME}s, before querying again.")
            time.sleep(WAIT_TIME)

        logger.info("Server Started")
        client = Client(LOCAL_HOST_URI, timeout=client_timeout)  # use deployment settings
        logger.info(f"Created Client: {client}")
    except Exception as e:
        raise Exception(f"Error in creating client or server: {e}")


"""
Read about client accepted parameters here:
    https://github.com/huggingface/text-generation-inference/tree/5a1512c0253e759fb07142029127292d639ab117/clients/python/text_generation

client.__init__():
    base_url: str,
    headers: Optional[Dict[str, str]] = None,
    cookies: Optional[Dict[str, str]] = None,
    timeout: int = 10,

client:generate():
    self,
    prompt: str,
    do_sample: bool = False,
    max_new_tokens: int = 20,
    best_of: Optional[int] = None,
    repetition_penalty: Optional[float] = None,
    return_full_text: bool = False,
    seed: Optional[int] = None,
    stop_sequences: Optional[List[str]] = None,
    temperature: Optional[float] = None,
    top_k: Optional[int] = None,
    top_p: Optional[float] = None,
    truncate: Optional[int] = None,
    typical_p: Optional[float] = None,
    watermark: bool = False,
    decoder_input_details: bool = False,

class Parameters(BaseModel):
    # Activate logits sampling
    do_sample: bool = False
    # Maximum number of generated tokens
    max_new_tokens: int = 20
    # The parameter for repetition penalty. 1.0 means no penalty.
    # See [this paper](https://arxiv.org/pdf/1909.05858.pdf) for more details.
    repetition_penalty: Optional[float] = None
    # Whether to prepend the prompt to the generated text
    return_full_text: bool = False
    # Stop generating tokens if a member of `stop_sequences` is generated
    stop: List[str] = []
    # Random sampling seed
    seed: Optional[int]
    # The value used to module the logits distribution.
    temperature: Optional[float]
    # The number of highest probability vocabulary tokens to keep for top-k-filtering.
    top_k: Optional[int]
    # If set to < 1, only the smallest set of most probable tokens with probabilities that add up to `top_p` or
    # higher are kept for generation.
    top_p: Optional[float]
    # truncate inputs tokens to the given size
    truncate: Optional[int]
    # Typical Decoding mass
    # See [Typical Decoding for Natural Language Generation](https://arxiv.org/abs/2202.00666) for more information
    typical_p: Optional[float]
    # Generate best_of sequences and return the one if the highest token logprobs
    best_of: Optional[int]
    # Watermarking with [A Watermark for Large Language Models](https://arxiv.org/abs/2301.10226)
    watermark: bool = False
    # Get generation details
    details: bool = False
    # Get decoder input token logprobs and ids
    decoder_input_details: bool = False
"""


"""
{
   "inputs": {
      "input_string":"the meaning of life is",
      "parameters":{
        "max_new_tokens":100,
        "do_sample": True
      }
   }
}

or 

{
   "inputs": "the meaning of life is",
    "parameters": {
        "max_new_tokens":200,
        "do_sample": True
    }
}

{
	"inputs": "User: Write an elaborate story about a hare and a tortoise that ends with a moral. The story should be set on the beautiful landscape of Bahamas.\nAssistant:",
	"parameters": {
		"max_new_tokens":500
	}
}
"""
def get_request_data(request_string):
    try:
        data = json.loads(request_string)
        logger.info(f"data: {data}")
        inputs = data["inputs"]
        # support both structure for now
        if isinstance(inputs, dict):
            input_data = inputs["input_string"]
            params = inputs.get("parameters", {})
        elif isinstance(inputs, str):
            input_data = inputs
            params = data.get("parameters", {})
        if not isinstance(input_data, str):
            raise Exception("input_str is not a str")
        if not isinstance(params, dict):
            raise Exception("parameters is not a dict")
        return input_data, params
    except Exception as e:
        raise Exception(json.dumps({
            "error": "expected input in format {'inputs': {'input_string': 'query', 'parameters': {k1=v1, k2=v2}}}",
            "exception": str(e)
        }))


def run(data):
    global client

    try:
        if client is None:
            raise Exception("Client is not initialized")

        query, params = get_request_data(data)
        logger.info(f"input_string: {query}, parameters: {params}")

        logger.info(f"generating response")
        time_start = time.time()
        response_str = client.generate(query, **params).generated_text
        time_taken = time.time() - time_start

        result_dict = {'responses': f'{response_str}', 'time': time_taken}
        logger.info(result_dict)
        return json.dumps(result_dict)
    except Exception as e:
        return json.dumps({
            "error": "Error in processing request",
            "exception": str(e)
        })


configs = get_init_args()
logger.info(configs)


if __name__ == "__main__":
    logger.info(init())
    logger.info(run({
        "inputs":{
            "input_string":"the meaning of life is",
            "parameters":{"max_new_tokens": 100, "do_sample": True}
        }
    }))
    logger.info(run({
        "inputs":"the meaning of life is",
        "parameters":{"max_new_tokens": 100, "do_sample": True}
    }))
