import os
import shutil
from mldesigner import command_component, Input, Output


@command_component()
def train_model(input_model: Input, output_model: Output):
    print(f"Training the input model.")
    for item in os.listdir(input_model):
        src = os.path.join(input_model, item)
        dst = os.path.join(output_model, item)
        if os.path.isdir(src):
            shutil.copytree(src, dst)
        else:
            shutil.copy2(src, dst)
