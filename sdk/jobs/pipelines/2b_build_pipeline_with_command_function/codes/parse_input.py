# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------

import argparse
import csv


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--input',
        type=str,
        help='Input file to parse.',
    )
    parser.add_argument(
        '--output',
        type=str,
        help='Parse result.',
    )
    parser.add_argument(
        '--column_name',
        type=str,
        help='Column name to filter.'
    )
    args, _ = parser.parse_known_args()

    input_path = args.input
    output_path = args.output
    column_name = args.column_name

    print(f"Received parameters:")
    print(f"{input_path} {output_path} {column_name}")

    with open(input_path) as csv_input:
        with open(output_path, 'w') as csv_output:
            reader = csv.DictReader(csv_input)
            try:
                header = next(reader)
            except Exception as e:
                raise Exception("Failed to load csv header.") from e
            writer = csv.DictWriter(csv_output, fieldnames=header)
            writer.writeheader()
            for row in reader:
                val = row.get(column_name, None)
                if val is not None and val != "":
                    writer.writerow(row)
