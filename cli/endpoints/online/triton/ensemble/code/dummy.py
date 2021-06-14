import json
import numpy as np


def init():
    print("Test")


def run(raw_data):
    np.array(json.loads(raw_data)["data"])
    # you can return any data type as long as it is JSON-serializable
    return ["Test"]
