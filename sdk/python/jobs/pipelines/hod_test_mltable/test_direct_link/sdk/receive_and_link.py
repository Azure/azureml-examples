import os
from pathlib import Path

from mldesigner import command_component, Input, Output
# from mldesigner.dsl._dynamic_executor import DynamicExecutor


@command_component
def gen_data_0(output: Output(type="uri_folder")):
    """Generate data output"""
    
    print(f"output_0: {output}")
    output_file = Path(output) / "hello.txt"
    output_file.write_text("Hello")
    

@command_component
def gen_data_1(output: Output(type="uri_folder")):
    """Generate data output"""
    
    print(f"output_1: {output}")
    output_file = Path(output) / "world.txt"
    output_file.write_text("Hello")
    

@command_component
def select_data(input_0: Input(type="uri_folder"), input_1: Input(type="uri_folder"), output: Output(type="mltable")):
    """Select input and gen mltable output"""

    print("============================ Inspect original inputs ============================")
    print(f"Input_0: {input_0}")
    print(f"Input_1: {input_1}")
    print(f"output: {output}")

    print("================= Write Mltable to local =================")
    mltable_yaml = f"""$schema: https://azuremlschemas.azureedge.net/latest/MLTable.schema.json

type: mltable
paths:
    - folder: {input_0}
"""

    print("Writing MLTable file to local...")
    print(f"The mltable_yaml: \n{mltable_yaml}")
    mltable = Path(output) / "MLTable"
    mltable.write_text(mltable_yaml)
    print("Mltable file ready.")
    
    print("================= Double check mltable file =================")
    print(f"mltable path: {mltable}")
    print(f"mltable content: \n{mltable.read_text()}")

    print("============================ Finished ============================")


@command_component
def consume_data(data_path: Input(type="mltable", mode="eval_mount")):
    """Consume an mltable input"""
    data = Path(data_path)
    is_folder = data.is_dir()
    print(f"Data path is a folder: {is_folder}")
    print(f"Data path: {data}")
    if is_folder:
        print("List all contents in data folder:")
        list_files(str(data))
    else:
        print(f"Data file name: {data.name}")
        print(f"Data file content: \n\t{data.read_text()}")


def list_files(startpath):
    for root, dirs, files in os.walk(startpath):
        level = root.replace(startpath, '').count(os.sep)
        indent = ' ' * 4 * (level)
        print('{}{}/'.format(indent, os.path.basename(root)))
        subindent = ' ' * 4 * (level + 1)
        for f in files:
            print('{}{}'.format(subindent, f))