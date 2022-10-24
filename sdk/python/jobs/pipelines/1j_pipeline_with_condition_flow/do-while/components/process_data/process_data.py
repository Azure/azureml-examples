import os
import shutil
from mldesigner import command_component, Input, Output


@command_component()
def process_data(input_data: Input, output_data: Output):
    print(f"Process the input data.")
    for item in os.listdir(input_data):
        src = os.path.join(input_data, item)
        dst = os.path.join(output_data, item)
        if os.path.isdir(src):
            shutil.copytree(src, dst)
        else:
            shutil.copy2(src, dst)
