"""score_bidaf.py

Scoring script for use with the Bi-directional Attention Flow model from the ONNX model zoo.
https://github.com/onnx/models/tree/master/text/machine_comprehension/bidirectional_attention_flow
"""

import json
import nltk
import numpy as np
import os

from nltk import word_tokenize
from utils import get_model_info, parse_model_http, triton_init, triton_infer
from tritonclientutils import triton_to_np_dtype


def preprocess(text, dtype):
    """Tokenizes text for use in the bidirectional attention flow model

    Parameters
    ---------
    text : str
        Text to be tokenized

    dtype : numpy datatype
        Datatype of the resulting array

    Returns
    ---------
    (np.array(), np.array())
        Tuple containing two numpy arrays with the tokenized
        words and chars, respectively.
    
    From: https://github.com/onnx/models/tree/master/text/machine_comprehension/bidirectional_attention_flow  # noqa
    """
    tokens = word_tokenize(text)
    # split into lower-case word tokens, in numpy array with shape of (seq, 1)
    words = np.array([w.lower() for w in tokens], dtype=dtype).reshape(-1, 1)
    # split words into chars, in numpy array with shape of (seq, 1, 1, 16)
    chars = [[c for c in t][:16] for t in tokens]
    chars = [cs+['']*(16-len(cs)) for cs in chars]
    chars = np.array(chars, dtype=dtype).reshape(-1, 1, 1, 16)
    return words, chars


def postprocess(context_words, answer):
    """Post-process results to show the chosen result

    Parameters
    --------
    context_words : str
        Original context

    answer : InferResult
        Triton inference result containing start and
        end positions of desired answer

    Returns
    --------
    Numpy array containing the words from the context that
    answer the given query.
    """

    start = answer.as_numpy("start_pos")[0]
    end = answer.as_numpy("end_pos")[0]
    print(f'start is {start}, end is {end}')
    return [w.encode() for w in context_words[start:end+1].reshape(-1)]


def init(url):
    global triton_client
    triton_client = triton_init(url)
        
    nltk.download('punkt')
        
    print(get_model_info())


def run(request):
    """This function is called every time the webservice receives a request.

    Notice you need to know the names and data types of the model inputs and
    outputs. You can get these values by reading the model configuration file
    or by querying the model metadata endpoint (see parse_model_http in
    utils.py for an example of how to do this)

    Parameters
    ----------
    request : str
        A valid JSON-formatted string containing input data, should
        contain both the context and a query for the Bidirectional
        Attention Flow model

    Returns
    ----------
    result : str
        String representing the words that answer the provided query

    """

    print(f'request is {request} type is {type(request)}')
    model_name = "bidaf-9"

    request = json.loads(request)
    context = request[0]
    query = request[1]

    input_meta, _, _, _ = parse_model_http(
        model_name=model_name)

    # We use the np.object data type for string data
    np_dtype = triton_to_np_dtype(input_meta[0]["datatype"])
    cw, cc = preprocess(context, np_dtype)
    qw, qc = preprocess(query, np_dtype)

    input_mapping = {
        "query_word": qw,
        "query_char": qc,
        "context_word": cw,
        "context_char": cc
    }

    res = triton_infer(input_mapping=input_mapping, model_name=model_name)

    result = postprocess(context_words=cw, answer=res)
    return result
