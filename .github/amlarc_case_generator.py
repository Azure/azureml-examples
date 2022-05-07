import argparse
import ruamel.yaml as yaml


def convert(input_file):
    with open(input_file, 'r') as f:
        data = yaml.round_trip_load(f)
        # remove install az cli step
        steps = data['jobs']['build']['steps']
        new_step = []
        for i in steps:
            if i['name'] != 'install az cli':
                new_step.append(i)
        data['jobs']['build']['steps'] = new_step

        # modify the pull request trigger
        orig_paths = data['on']['pull_request']['paths']
        new_paths = []
        for i in orig_paths:
            if '.github/workflows/cli-jobs' in i:
                t = i.split('/')
                tt = t[-1].split('.')
                tt = '%s-amlarc.yml' % tt[0]
                new_paths.append('/'.join(t[:-1] + [tt]))
                continue
            if 'sh' in i:
                continue
            new_paths.append(i)
        new_paths.append('.github/amlarc-tool.sh')
        data['on']['pull_request']['paths'] = new_paths


    # write back
    with open(input_file, 'w') as f:
        yaml.round_trip_dump(data, f, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # input file
    parser.add_argument("-i", "--input", type=str, required=True, help="input file")
    args = parser.parse_args()
    convert(args.input)
