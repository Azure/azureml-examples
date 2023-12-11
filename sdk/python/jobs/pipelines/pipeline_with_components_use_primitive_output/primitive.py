import argparse


if __name__ == '__main__':
    parser = argparse.ArgumentParser()


    parser.add_argument(
        '--input-int',
        type=int,
    )
    parser.add_argument(
        '--input-float',
        type=float,
    )
    parser.add_argument(
        '--input-str',
        type=str,
    )
    parser.add_argument(
        '--input-bool',
        type=bool,
    )
    parser.add_argument(
        '--output-int',
        type=str,
    )
    parser.add_argument(
        '--output-float',
        type=str,
    )
    parser.add_argument(
        '--output-str',
        type=str,
    )
    parser.add_argument(
        '--output-bool',
        type=str,
    )

    args, _ = parser.parse_known_args()

    print('input_int: ', args.input_int)
    print('input_float: ', args.input_float)
    print('input_str: ', args.input_str)
    print('input_bool: ', args.input_bool)
    print('output_int: ', args.output_int)
    print('output_float: ', args.output_float)
    print('output_str: ', args.output_str)
    print('output_bool: ', args.output_bool)
    with open(args.output_int, 'w') as f:
        f.write(str(args.input_int))
    with open(args.output_float, 'w') as f:
        f.write(str(args.input_float))
    with open(args.output_str, 'w') as f:
        f.write(str(args.input_str))
    with open(args.output_bool, 'w') as f:
        f.write(str(args.input_bool))
