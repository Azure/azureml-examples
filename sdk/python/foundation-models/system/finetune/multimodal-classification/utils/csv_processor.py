import argparse
from os.path import basename
import pandas as pd

parser = argparse.ArgumentParser()

# Input file name
parser.add_argument("-img_col_name", "--img_col_name", help="Column name of column in .csv file which has path to images.")
parser.add_argument("-image_url_prefix", "--image_url_prefix", help="URL of datastore where images are uploaded.")
parser.add_argument("-file_name", "--file_name", default="", help=".csv file that has the data.")

args = parser.parse_args()


def update_img_url(img_col_name: str, image_url_prefix: str, file_name: str):
    """
    Load .csv file at path `file_name`,
    extract file name of image from path in column `img_col_name`,
    add `image_url_prefix` to the file name and update the column `img_col_name`.

    :param img_col_name:Column name in csv file that has path to images.
    :type img_col_name: str
    :param image_url_prefix: URL of datastore where images are uploaded.
    :type image_url_prefix: str
    :param file_name: .csv file that has the data.
    :type file_name: str

    :return: None
    """
    df = pd.read_csv(file_name)
    df[img_col_name] = df[img_col_name].apply(lambda x: image_url_prefix + basename(x))
    df.to_csv(file_name, index=False)


if __name__ == "__main__":
    update_img_url(args.img_col_name,
                   args.image_url_prefix,
                   args.file_name)
