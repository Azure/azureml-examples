import argparse
import pandas as pd

parser = argparse.ArgumentParser()

# Input file name
parser.add_argument(
    "-img_col_name",
    "--img_col_name",
    help="Column name of column in .csv file which has path to images.",
)
parser.add_argument(
    "-image_url_prefix",
    "--image_url_prefix",
    help="URL of datastore where images are uploaded.",
)
parser.add_argument(
    "-input_file_name",
    "--input_file_name",
    default="",
    help=".csv file that has the data.",
)
parser.add_argument(
    "-output_file_name",
    "--output_file_name",
    default="",
    help=".csv file that has the formatted data.",
)

args = parser.parse_args()


def update_img_url(
    img_col_name: str,
    image_url_prefix: str,
    input_file_name: str,
    output_file_name: str,
):
    """
    Load .csv file at path `file_name`,
    extract file name of image from path in column `img_col_name`,
    add `image_url_prefix` to the file name and update the column `img_col_name`.

    :param img_col_name:Column name in csv file that has path to images.
    :type img_col_name: str
    :param image_url_prefix: URL of datastore where images are uploaded.
    :type image_url_prefix: str
    :param input_file_name: Path to csv file.
    :type input_file_name: str
    :param output_file_name: Path to output csv file.
    :type output_file_name: str

    :return: None
    """
    df = pd.read_csv(input_file_name)
    df[img_col_name] = df[img_col_name].apply(
        lambda x: image_url_prefix + "/".join(x.strip("/").split("/")[-2:])
    )
    df.to_csv(output_file_name, index=False)
    print("Updated image urls in csv file. Saved at: ", output_file_name)


if __name__ == "__main__":
    update_img_url(
        args.img_col_name,
        args.image_url_prefix,
        args.input_file_name,
        args.output_file_name,
    )
