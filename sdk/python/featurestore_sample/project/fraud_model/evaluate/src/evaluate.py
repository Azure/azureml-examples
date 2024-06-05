import argparse
import os
import shutil

# This is a placeholder no-op file for illustration purpose only. Do not use it for production use.
print("inside evaluate")
parser = argparse.ArgumentParser("train")
parser.add_argument("--model_input", type=str, help="model_input")
parser.add_argument("--evaluation_output", type=str, help="evaluation_output")

args = parser.parse_args()

# User should fail this job (throw exception) if the evaluation result is not good
# That way the model will not be registered by the next step

f = open(args.evaluation_output + "/evaluation.txt", "x")
f.write("Now the file has more content!")
f.close()
