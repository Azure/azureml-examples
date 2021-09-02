import argparse
import torch
import model_info_lib

# Initiate the parser
parser = argparse.ArgumentParser()

parser.add_argument("--input", "-i", help="input pytorch_filename")
parser.add_argument("--output", "-o", help="output onnx_filename")

args = parser.parse_args()

if args.input:
    print("Converting from pytorch model at : %s" % args.input)
    pytorch_model_weights_filename = args.input
else:
    raise ValueError("--input pytorch_filename has to be specified")

if args.output:
    print("To ONNX model at : %s" % args.output)
    onnx_model_filename = args.output
else:
    raise ValueError("--output onnx_filename has to be specified")

model,model_input,original_input_names,original_output_names = model_info_lib.get_model_info()
model.load_state_dict(torch.load(pytorch_model_weights_filename))

if len(original_input_names) > 0 and  len(original_output_names) > 0:
    torch.onnx.export(
    model,
    model_input,
    onnx_model_filename,
    input_names=original_input_names,
    output_names=original_output_names,
    )
else:
    torch.onnx.export(model, model_input,onnx_model_filename) 