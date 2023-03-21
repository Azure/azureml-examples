import sys
import torch
import os
from collections import OrderedDict
from pathlib import Path

# insert megatron's root dir into sys.path
root_repo_path = str(Path(__file__).resolve().parents[2])
if root_repo_path not in sys.path:
    sys.path.insert(0, root_repo_path)


def dump_data(datum, name_list=[]):
    if type(datum) in (dict, OrderedDict):
        for k, v in datum.items():
            dump_data(v, name_list + [str(k)])
    elif type(datum) in (list, tuple):
        for v in datum:
            dump_data(v, name_list)
    elif torch.is_tensor(datum):
        prefix = '.'.join(name_list)
        print(f'[tensor] {prefix} = {datum.shape}')
    else:
        #pass
        prefix = '.'.join(name_list)
        print(f'[other] {prefix} = {datum}')


def main():
    if len(sys.argv) < 2:
        print(f'Usage: {sys.argv[0]} <checkpoint file>')
        exit(1)

    ckpt_file = sys.argv[1]
    if not os.path.isfile(ckpt_file):
        print(f'{ckpt_file} is not a valid file')
        exit(1)

    print(f'loading checkpoint file: {ckpt_file}')
    sd = torch.load(ckpt_file, map_location=torch.device('cpu'))
    dump_data(sd)

    quit()


if __name__ == "__main__":
    main()
