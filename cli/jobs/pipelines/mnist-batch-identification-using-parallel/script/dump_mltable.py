import yaml
import argparse
import os

d = {"paths": [{"folder": "./mnist"}]}

parser = argparse.ArgumentParser(allow_abbrev=False, description="dump mltable")

parser.add_argument("--output_folder", type=str, default=0)
args, _ = parser.parse_known_args()
dump_path = os.path.join(args.output_folder, "MLTable")
with open(dump_path, "w") as yaml_file:
    yaml.dump(d, yaml_file, default_flow_style=False)
print("Saved MLTable file")
