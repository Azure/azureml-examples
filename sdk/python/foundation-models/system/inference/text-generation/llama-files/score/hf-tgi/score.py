# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------

import asyncio
import json
import os
import psutil
import requests
import time
import yaml
import logging
import pandas as pd
import numpy as np

from concurrent.futures import ThreadPoolExecutor
from mlflow.pyfunc.scoring_server import _get_jsonable_obj
from typing import List, Dict, Any, Tuple, Union
from text_generation import Client

from azure.ai.contentsafety import ContentSafetyClient
from azure.core.credentials import AzureKeyCredential
from azure.ai.contentsafety.models import AnalyzeTextOptions
from aiolimiter import AsyncLimiter
from azure.core.pipeline.policies import HeadersPolicy


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

# AACS
aacs_threshold = int(os.environ.get("CONTENT_SAFETY_THRESHOLD", 2))


class SupportedTask:
    """Supported tasks by text-generation-inference"""
    TEXT_GENERATION = "text-generation"
    CHAT_COMPLETION = "chat-completion"


# default values
MLMODEL_PATH = "mlflow_model_folder/MLmodel"
DEFAULT_MODEL_ID_PATH  = "mlflow_model_folder/data/model"
client = None
task_type = SupportedTask.TEXT_GENERATION


def is_server_healthy():
    """Periodically checks if server is up and running."""
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

# ACS START


class AsyncRateLimitedOpsUtils:
    # 1000 requests / 10 seconds. Limiting to 800 request per 10 secods
    # limiting to 1000 concurrent requests
    def __init__(
        self,
        ops_count=800,
        ops_seconds=10,
        concurrent_ops=1000,
        thread_max_workers=1000,
    ):
        self.limiter = AsyncLimiter(ops_count, ops_seconds)
        self.semaphore = asyncio.Semaphore(value=concurrent_ops)
        # need thread pool executor for sync function
        self.executor = ThreadPoolExecutor(max_workers=thread_max_workers)

    def get_limiter(self):
        return self.limiter

    def get_semaphore(self):
        return self.semaphore

    def get_executor(self):
        return self.executor


async_rate_limiter = AsyncRateLimitedOpsUtils()


class CsChunkingUtils:
    def __init__(self, chunking_n=1000, delimiter="."):
        self.delimiter = delimiter
        self.chunking_n = chunking_n

    def chunkstring(self, string, length):
        return (string[0 + i : length + i] for i in range(0, len(string), length))

    def split_by(self, input):
        max_n = self.chunking_n
        split = [e + self.delimiter for e in input.split(self.delimiter) if e]
        ret = []
        buffer = ""

        for i in split:
            # if a single element > max_n, chunk by max_n
            if len(i) > max_n:
                ret.append(buffer)
                ret.extend(list(self.chunkstring(i, max_n)))
                buffer = ""
                continue
            if len(buffer) + len(i) <= max_n:
                buffer = buffer + i
            else:
                ret.append(buffer)
                buffer = i

        if len(buffer) > 0:
            ret.append(buffer)
        return ret

async def async_analyze_text_task(client, request):
    loop = asyncio.get_event_loop()
    executor = async_rate_limiter.get_executor()
    sem = async_rate_limiter.get_semaphore()
    await sem.acquire()
    async with async_rate_limiter.get_limiter():
        response = await loop.run_in_executor(executor, client.analyze_text, request)
        sem.release()
        severity = analyze_response(response)
        return severity


def analyze_response(response):
    severity = 0

    if response.hate_result is not None:
        print("Hate severity: {}".format(response.hate_result.severity))
        severity = max(severity, response.hate_result.severity)
    if response.self_harm_result is not None:
        print("SelfHarm severity: {}".format(response.self_harm_result.severity))
        severity = max(severity, response.self_harm_result.severity)
    if response.sexual_result is not None:
        print("Sexual severity: {}".format(response.sexual_result.severity))
        severity = max(severity, response.sexual_result.severity)
    if response.violence_result is not None:
        print("Violence severity: {}".format(response.violence_result.severity))
        severity = max(severity, response.violence_result.severity)

    return severity


def analyze_text_async(text):
    # Chunk text
    chunking_utils = CsChunkingUtils(chunking_n=1000, delimiter=".")
    split_text = chunking_utils.split_by(text)

    tasks = []
    for i in split_text:
        request = AnalyzeTextOptions(text=i)
        tasks.append(async_analyze_text_task(aacs_client, request))

    done, pending = asyncio.get_event_loop().run_until_complete(
        asyncio.wait(tasks, timeout=60)
    )

    if len(pending) > 0:
        # not all task finished, assume failed
        return 6

    return max([d.result() for d in done])


def analyze_text(text):
    # Chunk text
    print(f"Analyzing ...")
    chunking_utils = CsChunkingUtils(chunking_n=1000, delimiter=".")
    split_text = chunking_utils.split_by(text)

    result = [
        analyze_response(aacs_client.analyze_text(AnalyzeTextOptions(text=i)))
        for i in split_text
    ]
    severity = max(result)
    print(f"Analyzed, severity {severity}")

    return severity


def iterate(obj):
    if isinstance(obj, dict):
        severity = 0
        for key, value in obj.items():
            obj[key], value_severity = iterate(value)
            severity = max(severity, value_severity)
        return obj, severity
    elif isinstance(obj, list) or isinstance(obj, np.ndarray):
        severity = 0
        for idx in range(len(obj)):
            obj[idx], value_severity = iterate(obj[idx])
            severity = max(severity, value_severity)
        return obj, severity
    elif isinstance(obj, pd.DataFrame):
        severity = 0
        for i in range(obj.shape[0]):  # iterate over rows
            for j in range(obj.shape[1]):  # iterate over columns
                obj.at[i, j], value_severity = iterate(obj.at[i, j])
                severity = max(severity, value_severity)
        return obj, severity
    elif isinstance(obj, str):
        severity = analyze_text(obj)
        if severity > aacs_threshold:
            return "", severity
        else:
            return obj, severity
    else:
        return obj, 0


def get_safe_response(result):
    print("Analyzing response...")
    jsonable_result = _get_jsonable_obj(result, pandas_orient="records")

    result, severity = iterate(jsonable_result)
    print(f"Response analyzed, severity {severity}")
    return result


def get_safe_input(input_data):
    print("Analyzing input...")
    result, severity = iterate(input_data)
    print(f"Input analyzed, severity {severity}")
    return result, severity

# ACS END


def init():
    """Initialize text-generation-inference server and client."""
    global client
    global task_type
    global aacs_client

    try:
        logger.info("Setting up AACS")
        endpoint = os.environ.get("CONTENT_SAFETY_ENDPOINT", None)
        key = os.environ.get("CONTENT_SAFETY_KEY", None)

        if not endpoint:
            raise Exception("CONTENT_SAFETY_ENDPOINT env not set for AACS.")
        if not key:
            raise Exception("CONTENT_SAFETY_KEY env not set for AACS.")

        # Create an Content Safety client
        headers_policy = HeadersPolicy()
        headers_policy.add_header("ms-azure-ai-sender", "llama")
        aacs_client = ContentSafetyClient(
            endpoint, AzureKeyCredential(key), headers_policy=headers_policy
        )
    except Exception as e:
        raise Exception("Error in setting up AACS client. Error  {e}")

    try:
        model_id = os.environ.get(MODEL_ID, DEFAULT_MODEL_ID_PATH)
        client_timeout = os.environ.get(CLIENT_TIMEOUT, MAX_REQUEST_TIMEOUT)

        for k, v in os.environ.items():
            logger.info(f"env: {k} = {v}")

        model_path = os.path.join(os.getenv("AZUREML_MODEL_DIR", ""), model_id)
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

        # run nvidia-smi
        logger.info("###### GPU INFO ######")
        logger.info(os.system("nvidia-smi"))
        logger.info("###### GPU INFO ######")

        client = Client(LOCAL_HOST_URI, timeout=client_timeout)  # use deployment settings
        logger.info(f"Created Client: {client}")
    except Exception as e:
        raise Exception(f"Error in creating client or server: {e}")


def get_processed_input_data_for_chat_completion(data: List[str]) -> str:
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
    B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"
    DEFAULT_SYSTEM_PROMPT = """\
    You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.

    If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information."""

    dialog = data
    history = ""
    assert len(dialog) > 0
    # add a system prompt if the first message is not a system prompt
    if dialog[0]["role"] != "system":
        dialog = [
            {
                "role": "system",
                "content": DEFAULT_SYSTEM_PROMPT,
            }
        ] + dialog
    # add a tag to the system prompt
    dialog = [
        {
            "role": dialog[0]["role"],
            "content": B_SYS + dialog[0]["content"] + E_SYS
            # + dialog[1]["content"],
        }
    ] + dialog[1:]
    assert all([msg["role"] == "user" for msg in dialog[1::2]]) and all(
        [msg["role"] == "assistant" for msg in dialog[2::2]]
    ), (
        "model only supports 'system', 'user' and 'assistant' roles, "
        "starting with 'system', then 'user' and alternating (u/a/u/a/u...)"
    )
    history += dialog[0]["content"]
    assert dialog[-1]["role"] == "user"
    for i, conv in enumerate(dialog[1:]):
        if i % 2 == 1:
            assert conv["role"] == "assistant"
            history += f"{conv['content'].strip()}"
        else:
            assert conv["role"] == "user"
            history += B_INST + f"{conv['content'].strip()}" + E_INST
    return history


def get_request_data(request_string) -> Tuple[Union[str, List[str]], Dict[str, Any]]:
    """
    return type for chat-completion: str, dict
    return type for text-generation: list, dict
    """
    global task_type
    try:
        data = json.loads(request_string)
        logger.info(f"data: {data}")
        inputs = data.get("input_data", None)

        input_data = []   # type: Union[str, List[str]]
        params = {} # type: Dict[str, Any]

        if not isinstance(inputs, dict):
            raise Exception("Invalid input data")

        input_data = inputs["input_string"]
        params = inputs.get("parameters", {})

        if not isinstance(input_data, list):
            raise Exception("query is not a list")

        if not isinstance(params, dict):
            raise Exception("parameters is not a dict")

        if task_type == SupportedTask.CHAT_COMPLETION:
            print("chat-completion task. Processing input data")
            input_data = get_processed_input_data_for_chat_completion(input_data)

        return input_data, params
    except Exception as e:
        raise Exception(json.dumps({
            "error": (
                'Expected input format: \n'
                '{"input_data": {"input_string": "<query>", "parameters": {"k1":"v1", "k2":"v2"}}}.\n '
                '<query> should be in below format:\n '
                'For text-generation: ["str1", "str2", ...]\n'
                'For chat-completion : [{"role": "user", "content": "str1"}, {"role": "assistant", "content": "str2"} ....]'
            ),
            "exception": str(e)
        }))


def run(data):
    """Run for inference data provided."""
    global client
    global task_type

    try:
        data, severity = get_safe_input(data)
        if severity > aacs_threshold:
            logger.warning(f"Input severity ({severity}) greater than aacs threshold ({aacs_threshold}).")
            return {}

        if client is None:
            raise Exception("Client is not initialized")

        query, params = get_request_data(data)
        logger.info(f"generating response for input_string: {query}, parameters: {params}")

        if task_type == SupportedTask.CHAT_COMPLETION:
            time_start = time.time()
            response_str = client.generate(query, **params).generated_text
            time_taken = time.time() - time_start
            logger.info(f"time_taken: {time_taken}")
            result_dict = {'output': f'{response_str}'}
            resp = pd.DataFrame([result_dict])
            return get_safe_response(resp)

        assert task_type == SupportedTask.TEXT_GENERATION and isinstance(query, list), "query should be a list for text-generation"

        results = {}
        for i, q in enumerate(query):
            time_start = time.time()
            response_str = client.generate(q, **params).generated_text
            time_taken = time.time() - time_start
            logger.info(f"query {i} - time_taken: {time_taken}")
            results[str(i)] = [f'{response_str}']

        resp = pd.DataFrame(results)
        return get_safe_response(resp)

    except Exception as e:
        return json.dumps({
            "error": "Error in processing request",
            "exception": str(e)
        })


if __name__ == "__main__":
    logger.info(init())
    assert task_type is not None

    valid_inputs = {
        "text-generation": [
            {
                "input_data":{
                    "input_string": ["the meaning of life is"],
                    "parameters":{"max_new_tokens": 100, "do_sample": True}
                }
            }
        ],
        "chat-completion": [
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
                        "temperature": 0.9,
                        "top_p": 0.6,
                        "do_sample": True,
                        "max_new_tokens":100 
                    }
                } 
            }
        ]
    }

    for sample_ip in valid_inputs[task_type]:
        logger.info(run(json.dumps(sample_ip)))
