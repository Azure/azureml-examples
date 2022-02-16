# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""
This script parses COCO annotations for validation and training image datasets.
Given a category id from COCO, it outputs a csv file containing each image id,
file name and a bool indicating if an object of that category can be found in the image.
"""
import os
import sys
import argparse
import logging
import json
import glob

def get_images_containing_category(
    annotations: dict, category_id: int, category_name: str
) -> dict:
    """Parses COCO annotations json structure to get
    images that have an object from a given category.

    Args:
        annotations (dict): json nested dictionary from annotations files
        category_id (int): identifier of a COCO annotation category (ex: 1 == 'person')
        category_name (str): readable name for that category

    Returns:
        image_id_selection (dict): dict containing file_name (str) and category_name (bool)
    """
    # first loop through images to get their ids
    image_id_selection = dict(
        [
            (entry["id"], {"file_name": entry["file_name"], category_name: False})
            for entry in annotations["images"]
        ]
    )

    # second loop through annotations
    for entry in annotations["annotations"]:
        if entry["image_id"] in image_id_selection:
            # capture if there's an annotation from that category inside the image
            if entry["category_id"] == category_id:
                image_id_selection[entry["image_id"]][category_name] = True
        else:
            logging.warning(
                f"Image id {entry['image_id']} from annotations is not found in list of image ids from images."
            )

    return image_id_selection


def save_image_list_as_csv(
    image_id_selection: dict,
    category_name,
    image_file_name_prefix: str,
    output_file_name: str,
):
    """Saves a list of images with a given category into a csv file.

    Args:
        image_id_selection (dict): structure provided by get_images_containing_category()
        category_name (str): readable name for the image category
        image_file_name_prefix (str): a prefix (folder?) applied to all image file names
        output_file_name (str): path to write csv file

    Returns:
        None
    """
    with open(output_file_name, "w") as out_file:
        for key in image_id_selection:
            out_file.write(
                "{0},{1}{2},{3}\n".format(
                    key,
                    image_file_name_prefix,
                    image_id_selection[key]["file_name"],
                    category_name
                    if image_id_selection[key][category_name]
                    else "not_" + category_name,
                )
            )


def run(args: argparse.Namespace):
    """Runs the script based on CLI arguments.

    Args:
        args (argparse.Namespace): arguments parsed from argparse

    Returns:
        None
    """
    # initialize root logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    console_handler = logging.StreamHandler()
    formatter = logging.Formatter(
        "%(asctime)s : %(levelname)s : %(name)s : %(message)s"
    )
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    logger.info(f"Running with arguments: {args}")

    # list all files in annotations folder

    #########################
    ### VALID ANNOTATIONS ###
    #########################

    # it's likely the file will be in some subfolder
    valid_file_path = glob.glob(os.path.join(args.annotations_dir, "**", "instances_val2017.json"), recursive=True)[0]

    logger.info(f"Loading valid annotations from {valid_file_path}")
    with open(valid_file_path) as in_file:
        valid_annotations = json.loads(in_file.read())
    logger.info(
        f"Loaded annotations from valid: found {len(valid_annotations['images'])} images"
    )

    # parses annotations to detect if image has annotation of a given category
    image_id_selection = get_images_containing_category(
        valid_annotations, args.category_id, args.category_name
    )

    # save the result as csv
    save_image_list_as_csv(
        image_id_selection,
        args.category_name,
        "val2017/",
        os.path.join(args.output_valid, "valid_annotations.txt"),
    )

    #########################
    ### TRAIN ANNOTATIONS ###
    #########################

    # it's likely the file will be in some subfolder
    train_file_path = glob.glob(os.path.join(args.annotations_dir, "**", "instances_train2017.json"), recursive=True)[0]

    logger.info(f"Loading train annotations from {train_file_path}")
    with open(train_file_path) as in_file:
        train_annotations = json.loads(in_file.read())
    logger.info(
        f"Loaded annotations from train: found {len(train_annotations['images'])} images"
    )

    # parses annotations to detect if image has annotation of a given category
    image_id_selection = get_images_containing_category(
        train_annotations, args.category_id, args.category_name
    )

    # save the result as csv
    save_image_list_as_csv(
        image_id_selection,
        args.category_name,
        "train2017/",
        os.path.join(args.output_train, "train_annotations.txt"),
    )

    logger.info("Closing.")


def main():
    """Main function of the script."""
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--annotations_dir",
        required=True,
        type=str,
        help="the path to annotations files from coco",
    )
    parser.add_argument(
        "--category_id",
        required=True,
        type=int,
        help="identifier of the category to extract",
    )
    parser.add_argument(
        "--category_name",
        required=True,
        type=str,
        help="readable name of the category",
    )
    parser.add_argument(
        "--output_train",
        required=True,
        type=str,
        help="path to output train annotations",
    )
    parser.add_argument(
        "--output_valid",
        required=True,
        type=str,
        help="path to output valid annotations",
    )

    args = parser.parse_args()

    run(args)


if __name__ == "__main__":
    main()
