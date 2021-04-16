"""score_bidaf.py

Scoring script for use with the Bi-directional Attention Flow model from the ONNX model zoo.
https://github.com/onnx/models/tree/master/text/machine_comprehension/bidirectional_attention_flow
"""

import json
import nltk
import numpy as np
import os

from nltk import word_tokenize
from .utils import get_model_info, parse_model_http, triton_init, triton_infer
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
    nltk.download("punkt")
    tokens = word_tokenize(text)
    # split into lower-case word tokens, in numpy array with shape of (seq, 1)
    words = np.array([w.lower() for w in tokens], dtype=dtype).reshape(-1, 1)
    # split words into chars, in numpy array with shape of (seq, 1, 1, 16)
    chars = [[c for c in t][:16] for t in tokens]
    chars = [cs + [""] * (16 - len(cs)) for cs in chars]
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
    print(f"start is {start}, end is {end}")
    return [w.encode() for w in context_words[start : end + 1].reshape(-1)]
