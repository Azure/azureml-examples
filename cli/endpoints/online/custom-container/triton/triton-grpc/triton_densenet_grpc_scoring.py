import argparse
import numpy as np
import os
from PIL import Image

import grpc
from tritonclient.grpc import service_pb2, service_pb2_grpc

def preprocess(img_content):
    """Pre-process an image to meet the size, type and format
    requirements specified by the parameters.
    """
    c = 3
    h = 224
    w = 224

    img = Image.open(img_content)

    sample_img = img.convert("RGB")

    resized_img = sample_img.resize((w, h), Image.BILINEAR)
    resized = np.array(resized_img)
    if resized.ndim == 2:
        resized = resized[:, :, np.newaxis]

    typed = resized.astype(np.float32)

    # scale for INCEPTION
    scaled = (typed / 128) - 1

    # Swap to CHW
    ordered = np.transpose(scaled, (2, 0, 1))

    # Channels are in RGB order. Currently model configuration data
    # doesn't provide any information as to other channel orderings
    # (like BGR) so we just assume RGB.
    img_array = np.array(ordered, dtype=np.float32)[None, ...]

    return img_array


def postprocess(max_label):
    """Post-process results to show the predicted label."""

    absolute_path = os.path.abspath(__file__)
    folder_path = os.path.dirname(absolute_path)
    label_path = os.path.join(folder_path, "densenet_labels.txt")
    print(label_path)

    label_file = open(label_path, "r")
    labels = label_file.read().split("\n")
    label_dict = dict(enumerate(labels))
    final_label = label_dict[max_label]
    return f"{max_label} : {final_label}"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", type=str, default="localhost:8001")
    parser.add_argument("--token")
    parser.add_argument("--image_path", type=str, default="https://aka.ms/peacock-pic")
    args = parser.parse_args()

    # Create gRPC stub for communicating with the server
    scoring_uri = args.url[8:]+":443"
    print(scoring_uri)
    channel = grpc.secure_channel(scoring_uri, grpc.ssl_channel_credentials())
    grpc_stub = service_pb2_grpc.GRPCInferenceServiceStub(channel)
    metadata = [('authorization', f'Bearer {args.token}')]

    # Health
    try:
        request = service_pb2.ServerLiveRequest()
        response = grpc_stub.ServerLive(request, metadata=metadata)
        print("server {}".format(response))
    except Exception as ex:
        print(ex)

    request = service_pb2.ServerReadyRequest()
    response = grpc_stub.ServerReady(request, metadata=metadata)
    print("server {}".format(response))

    model_name = "model_1"
    model_version = "1"

    request = service_pb2.ModelReadyRequest(name=model_name, version=model_version)
    response = grpc_stub.ModelReady(request, metadata=metadata)
    print("model {}".format(response))

    # Metadata
    request = service_pb2.ServerMetadataRequest()
    response = grpc_stub.ServerMetadata(request, metadata=metadata)
    print("server metadata:\n{}".format(response))

    request = service_pb2.ModelMetadataRequest(name=model_name, version=model_version)
    model_metadata = grpc_stub.ModelMetadata(request, metadata=metadata)
    print("model metadata:\n{}".format(model_metadata))

    # Configuration
    request = service_pb2.ModelConfigRequest(name=model_name, version=model_version)
    model_config = grpc_stub.ModelConfig(request, metadata=metadata)
    print("model config:\n{}".format(model_config))

    # Infer
    request = service_pb2.ModelInferRequest()
    request.model_name = model_name
    request.model_version = model_version
    request.id = "my request id"

    img_data = preprocess(args.image_path)

    input = service_pb2.ModelInferRequest().InferInputTensor()
    input.name = model_metadata.inputs[0].name
    input.datatype = model_metadata.inputs[0].datatype
    input.shape.extend(img_data.shape)
    request.inputs.extend([input])

    output = service_pb2.ModelInferRequest().InferRequestedOutputTensor()
    output.name = "fc6_1"
    request.outputs.extend([output])

    #response = grpc_stub.ModelInfer(request, metadata=metadata)
    print("model infer:\n{}".format(response))