import os
from pathlib import Path

from mldesigner import command_component, Input, Output


@command_component(environment="./conda.yaml")
def get_my_data(output: Output(type="mltable")):
    """Get a data asset and pass to next component"""

    output_base = Path(output)
    print("================= Check output folder =================")
    print(f"Output is a folder: {output_base.is_dir()}")
    print(f"Output base: {output_base}")

    print("================= Write Mltable to local =================")
    # azureml://subscriptions/96aede12-2f73-41cb-b983-6d11a904839b/resourcegroups/hod-eastus2/workspaces/sdk_vnext_cli/datastores/workspaceblobstore/paths/LocalUpload/cc355250825d6284521e9ae14f3db123/src/
    # azureml://subscriptions/96aede12-2f73-41cb-b983-6d11a904839b/resourcegroups/hod-eastus2/workspaces/sdk_vnext_cli/datastores/workspaceblobstore/paths/LocalUpload/4c8f25cc30097999fb052b06bca2a561/data/
    mltable_yaml = f"""$schema: https://azuremlschemas.azureedge.net/latest/MLTable.schema.json

type: mltable
paths:
    - file: azureml://subscriptions/96aede12-2f73-41cb-b983-6d11a904839b/resourcegroups/hod-eastus2/workspaces/sdk_vnext_cli/datastores/workspaceblobstore/paths/LocalUpload/4c8f25cc30097999fb052b06bca2a561/data/
"""

    print("Writing MLTable file to local...")
    print(f"The mltable_yaml: \n{mltable_yaml}")
    mltable = Path(output) / "MLTable"
    mltable.write_text(mltable_yaml)
    print("Mltable file ready.")
    
    print("================= Double check mltable file =================")
    print(f"mltable path: {mltable}")
    print(f"mltable content: \n{mltable.read_text()}")
    

@command_component(environment="./conda.yaml")
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