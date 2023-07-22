# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------

import json
import os
import psutil
import requests
import sys
import time
import yaml

import logging
from pathlib import Path
from subprocess import PIPE, STDOUT
from subprocess import run as subprocess_run
from typing import List, Dict, Any, Tuple, Union
from text_generation import Client


# Configure logger
logger = logging.getLogger(__name__)
format_str = "%(asctime)s [%(module)s] %(funcName)s %(lineno)s: %(levelname)-8s [%(process)d] %(message)s"
formatter = logging.Formatter(format_str)
stream_handler = logging.StreamHandler()
stream_handler.setFormatter(formatter)
logger.setLevel(logging.DEBUG)
logger.addHandler(stream_handler)

PORT = 80
LOCAL_HOST_URI = f"http://0.0.0.0:{PORT}"

TEXT_GEN_LAUNCHER_PROCESS_NAME = "text-generation-launcher"

# model init env vars
MODEL_ID = "MODEL_ID"
SHARDED = "SHARDED"
NUM_SHARD = "NUM_SHARD"
QUANTIZE = "QUANTIZE"
DTYPE = "DTYPE"
TRUST_REMOTE_CODE = "TRUST_REMOTE_CODE"
MAX_CONCURRENT_REQUESTS = "MAX_CONCURRENT_REQUESTS"
MAX_BEST_OF = "MAX_BEST_OF"
MAX_STOP_SEQUENCES = "MAX_STOP_SEQUENCES"
MAX_INPUT_LENGTH = "MAX_INPUT_LENGTH"
MAX_TOTAL_TOKENS = "MAX_TOTAL_TOKENS"

# client init env vars
CLIENT_TIMEOUT = "TIMEOUT"
MAX_REQUEST_TIMEOUT = 90  # 90s


class SupportedTask:
    """Supported tasks by text-generation-inference"""
    TEXT_GENERATION = "text-generation"
    CHAT_COMPLETION = "chat-completion"


# default values
MLMODEL_PATH = "mlflow_model_folder/MLmodel"
DEFAULT_MODEL_ID_PATH  = "mlflow_model_folder/data"
client = None
task_type = SupportedTask.TEXT_GENERATION


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


def is_server_healthy():
    # use psutil to go through active process 
    WAIT_TIME = 20
    RETRY_COUNT = 5
    count = 0
    while count < RETRY_COUNT and TEXT_GEN_LAUNCHER_PROCESS_NAME not in [p.name() for p in psutil.process_iter()]:
        logger.warning(f"Process {TEXT_GEN_LAUNCHER_PROCESS_NAME} is not running. Sleeping for {WAIT_TIME}s and retrying")
        time.sleep(WAIT_TIME)
        count += 1
    if count >= RETRY_COUNT:
        total_dur = RETRY_COUNT * WAIT_TIME
        raise Exception(f"Sever process not running after waiting for {total_dur}. Terminating")

    logger.info(f"Server process {TEXT_GEN_LAUNCHER_PROCESS_NAME} running. Hitting endpoint with 5s delay")
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
    global task_type

    try:
        model_id = os.environ.get(MODEL_ID, DEFAULT_MODEL_ID_PATH)
        client_timeout = os.environ.get(CLIENT_TIMEOUT, MAX_REQUEST_TIMEOUT)

        for k, v in os.environ.items():
            logger.info(f"env: {k} = {v}")

        model_path = os.path.join(os.getenv("AZUREML_MODEL_DIR", ""), model_id)
        model_path = os.path.join(model_path, "model")  # default model path for MLFlow model
        # Use safetensors if present
        if os.path.isdir(os.path.join(model_path, "safetensors")):
            model_path = os.path.join(model_path, "safetensors")
        
        abs_mlmodel_path = os.path.join(os.getenv("AZUREML_MODEL_DIR", ""), MLMODEL_PATH)
        mlmodel = {}
        if abs_mlmodel_path and os.path.exists(abs_mlmodel_path):
            with open(abs_mlmodel_path) as f:
                mlmodel = yaml.safe_load(f)

        if mlmodel:
            flavors = mlmodel.get("flavors", {})
            if "hftransformersv2" in flavors:
                task_type = flavors["hftransformersv2"]["task_type"]
                if task_type not in (SupportedTask.TEXT_GENERATION, SupportedTask.CHAT_COMPLETION):
                    raise Exception(f"Unsupported task_type {task_type}")

        logger.info(f"Loading model from path {model_path} for task_type: {task_type}")
        logger.info(f"List model_path = {os.listdir(model_path)}")

        logger.info("Starting server")
        cmd = f"text-generation-launcher --model-id {model_path} &"
        os.system(cmd)
        time.sleep(20)

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
   "input_data": {
      "input_string":"the meaning of life is",
      "parameters":{
        "max_new_tokens":100,
        "do_sample": True
      }
   }
}

or 

{
   "input_data": "the meaning of life is",
    "parameters": {
        "max_new_tokens":200,
        "do_sample": True
    }
}

{
	"input_data": "User: Write an elaborate story about a hare and a tortoise that ends with a moral. The story should be set on the beautiful landscape of Bahamas.\nAssistant:",
	"parameters": {
		"max_new_tokens":500
	}
}

for CC:
{ 
    "input_data": { 
        "input_string": [ 
            { 
                "role": "user", 
                "content": "What is the tallest building in the world?" 
            }, 
            { 
                "role": "assistant", 
                "content": "As of 2021, the Burj Khalifa in Dubai, United Arab Emirates is the tallest building in the world, standing at a height of 828 meters (2,722 feet). It was completed in 2010 and has 163 floors. The Burj Khalifa is not only the tallest building in the world but also holds several other records, such as the highest occupied floor, highest outdoor observation deck, elevator with the longest travel distance, and the tallest freestanding structure in the world." 
            }, 
            { 
                "role": "user", 
                "content": "and in Africa?" 
            }, 
            { 
                "role": "assistant", 
                "content": "In Africa, the tallest building is the Carlton Centre, located in Johannesburg, South Africa. It stands at a height of 50 floors and 223 meters (730 feet). The CarltonDefault Centre was completed in 1973 and was the tallest building in Africa for many years until the construction of the Leonardo, a 55-story skyscraper in Sandton, Johannesburg, which was completed in 2019 and stands at a height of 230 meters (755 feet). Other notable tall buildings in Africa include the Ponte City Apartments in Johannesburg, the John Hancock Center in Lagos, Nigeria, and the Alpha II Building in Abidjan, Ivory Coast" 
            }, 
            { 
                "role": "user", 
                "content": "and in Europe?" 
            } 
        ], 
        "parameters":{ 
            "max_length": 100,
            "temperature": 0.9,
            "top_p": 0.6,
            "do_sample": true,
            "max_new_tokens":100 
        } 
    } 
}

"""

def get_processed_input_data_for_cc(data: List[str]) -> str:
    """
    example input:
    [
        {"role": "user", "content": "What is the tallest building in the world?"},
        {"role": "assistant", "content": "As of 2021, the Burj Khalifa in Dubai"},
        {"role": "user", "content": "and in Africa?"},
    ]
    example output:
    "[INST]What is the tallest building in the world?[\INST]As of 2021, the Burj Khalifa in Dubai\n[INST]and in Africa?[/INST]"
    """
    B_INST, E_INST = "[INST]", "[/INST]"
    conv_arr = data
    history = ""
    assert len(conv_arr) > 0
    assert conv_arr[0]["role"] == "user"
    history += B_INST + conv_arr[0]["content"].strip() + E_INST
    assert conv_arr[-1]["role"] == "user"
    for i, conv in enumerate(conv_arr[1:]):
        if i % 2 == 0:
            assert conv["role"] == "assistant"
            history += conv["content"].strip() + "\n"
        else:
            assert conv["role"] == "user"
            history += B_INST + conv["content"].strip() + E_INST
    return history


def get_request_data(request_string) -> Tuple[Union[str, List[str]], Dict[str, Any]]:
    """
    return type for cc: str, dict
    return type for tg: list, dict
    """
    global task_type
    try:
        data = json.loads(request_string)
        logger.info(f"data: {data}")
        # hf-tgi expects "inputs", while mir inference payloads expect "input_data"
        inputs = data.get("input_data", data.get("inputs", None))

        input_data = []   # type: Union[str, List[str]]
        params = {} # type: Dict[str, Any]
        if isinstance(inputs, dict):
            input_data = inputs["input_string"]
            params = inputs.get("parameters", {})
        elif isinstance(inputs, str):
            input_data = inputs
            params = data.get("parameters", {})
        else:
            raise Exception("input_data is not a dict or string")

        if not isinstance(params, dict):
            raise Exception("parameters is not a dict")

        if task_type == SupportedTask.CHAT_COMPLETION:
            if not isinstance(input_data, list):
                raise Exception("input_str is not a list (for cc)")
            print("CC task. Processing input data")
            input_data = get_processed_input_data_for_cc(input_data)
        
        if task_type == SupportedTask.TEXT_GENERATION and isinstance(input_data, str):
            input_data = [input_data]

        return input_data, params
    except Exception as e:
        raise Exception(json.dumps({
            "error": (
                'Expected input format: \n'
                '{"input_data": {"input_string": "<query>", "parameters": {"k1":"v1", "k2":"v2"}}} or '
                '{"inputs": "<query>", "parameters": {"k1":"v1", "k2":"v2"}} \n'
                '<query> should be string for text-generation and for chat-completion a list in below format: \n'
                '[{"role": "user", "content": "str"}, {"role": "assistant", "content": "str"} ....]'
            ),
            "exception": str(e)
        }))


def run(data):
    global client
    global task_type

    try:
        if client is None:
            raise Exception("Client is not initialized")

        query, params = get_request_data(data)
        logger.info(f"generating response for input_string: {query}, parameters: {params}")

        if task_type == SupportedTask.CHAT_COMPLETION:
            time_start = time.time()
            response_str = client.generate(query, **params).generated_text
            time_taken = time.time() - time_start
            logger.info(f"time_taken: {time_taken}")
            result_dict = {'0': f'{response_str}'}
            return json.dumps(result_dict)
    
        assert task_type == SupportedTask.TEXT_GENERATION and isinstance(query, list), "query should be a list for text-generation"
        
        results = []
        for i, q in enumerate(query):
            time_start = time.time()
            response_str = client.generate(q, **params).generated_text
            time_taken = time.time() - time_start
            logger.info(f"query {i} - time_taken: {time_taken}")
            results.append({str(i): f'{response_str}'})
        return json.dumps(results)

    except Exception as e:
        return json.dumps({
            "error": "Error in processing request",
            "exception": str(e)
        })



if __name__ == "__main__":
    logger.info(init())
    # cc
    logger.info(run(json.dumps({ 
        "input_data": { 
            "input_string": [
                { 
                    "role": "user", 
                    "content": "What is the tallest building in the world?" 
                }, 
                { 
                    "role": "assistant", 
                    "content": "As of 2021, the Burj Khalifa in Dubai, United Arab Emirates is the tallest building in the world, standing at a height of 828 meters (2,722 feet). It was completed in 2010 and has 163 floors. The Burj Khalifa is not only the tallest building in the world but also holds several other records, such as the highest occupied floor, highest outdoor observation deck, elevator with the longest travel distance, and the tallest freestanding structure in the world." 
                }, 
                { 
                    "role": "user", 
                    "content": "and in Africa?" 
                }, 
                { 
                    "role": "assistant", 
                    "content": "In Africa, the tallest building is the Carlton Centre, located in Johannesburg, South Africa. It stands at a height of 50 floors and 223 meters (730 feet). The CarltonDefault Centre was completed in 1973 and was the tallest building in Africa for many years until the construction of the Leonardo, a 55-story skyscraper in Sandton, Johannesburg, which was completed in 2019 and stands at a height of 230 meters (755 feet). Other notable tall buildings in Africa include the Ponte City Apartments in Johannesburg, the John Hancock Center in Lagos, Nigeria, and the Alpha II Building in Abidjan, Ivory Coast" 
                }, 
                { 
                    "role": "user", 
                    "content": "and in Europe?" 
                } 
            ], 
            "parameters":{ 
                "temperature": 0.9,
                "top_p": 0.6,
                "do_sample": True,
                "max_new_tokens":100 
            }
        } 
    })))

    # # text gen
    # logger.info(run(json.dumps({
    #     "input_data":{
    #         "input_string":"the meaning of life is",
    #         "parameters":{"max_new_tokens": 100, "do_sample": True}
    #     }
    # })))
    # logger.info(run(json.dumps({
    #     "input_data":"the meaning of life is",
    #     "parameters":{"max_new_tokens": 100, "do_sample": True}
    # })))
