import json
import numpy as np
import os
import onnxruntime
from nltk import word_tokenize
import nltk


def init():
    nltk.download("punkt")
    global sess
    sess = onnxruntime.InferenceSession(
        os.path.join(os.getenv("AZUREML_MODEL_DIR"), "model.onnx")
    )


def run(request):
    print(request)
    text = json.loads(request)
    qw, qc = preprocess(text["query"])
    cw, cc = preprocess(text["context"])

    # Run inference
    test = sess.run(
        None,
        {"query_word": qw, "query_char": qc, "context_word": cw, "context_char": cc},
    )
    start = np.asscalar(test[0])
    end = np.asscalar(test[1])
    ans = [w for w in cw[start : end + 1].reshape(-1)]
    print(ans)
    return ans


def preprocess(word):
    tokens = word_tokenize(word)

    # split into lower-case word tokens, in numpy array with shape of (seq, 1)
    words = np.asarray([w.lower() for w in tokens]).reshape(-1, 1)

    # split words into chars, in numpy array with shape of (seq, 1, 1, 16)
    chars = [[c for c in t][:16] for t in tokens]
    chars = [cs + [""] * (16 - len(cs)) for cs in chars]
    chars = np.asarray(chars).reshape(-1, 1, 1, 16)
    return words, chars
