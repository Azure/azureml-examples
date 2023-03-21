import json
import argparse

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir',
        type=str,
        required=True,
        help='directory to save data'
        )
    args = parser.parse_args()
    return args

def main():
    args = get_args()

    for i in range(10):

        row_limit = 1000
        rows_to_save = [{'text': ''.join([str(i)+'-*']*128)}]

        with open('{}/dataset_{}.json'.format(args.dir, i), 'w') as f:
            f.write(
                '\n'.join(json.dumps(_i) for _i in rows_to_save*row_limit)
            )

if __name__ == '__main__':
    main()
