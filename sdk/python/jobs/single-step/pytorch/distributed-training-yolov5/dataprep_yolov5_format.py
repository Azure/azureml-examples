import os
import urllib.request as request
from zipfile import ZipFile
import argparse
import json
import numpy as np
import PIL.Image as Image
import xml.etree.ElementTree as ET
import glob
import random
import shutil

url = "https://cvbp-secondary.z19.web.core.windows.net/datasets/object_detection/odFridgeObjects.zip"

data_folder = "./yolov5/data"
print("data_folder found")
data_file = "odFridgeObjects.zip"
f_loc = data_folder + "/" + data_file.split(".")[0]

input_dir = "./yolov5/data/odFridgeObjects/annotations/"
output_dir = "./yolov5/data/odFridgeObjects/labels/"
image_dir = "./yolov5/data/odFridgeObjects/images/"

processed_folder = "datasets"
split_ratio = 0.8


def downloaddata(url, data_folder, data_file):
    os.makedirs("data", exist_ok=True)
    fname = data_folder + "/" + data_file
    # urllib.request.urlretrieve(url, filename=fname)
    request.urlretrieve(url, filename=fname)
    with ZipFile(fname, "r") as zip:
        print("extracting files...")
        zip.extractall(path=data_folder)
        print("done")
    # os.remove(data_file)


# ************************** xml2yolo from https://gist.github.com/wfng92/c77c822dad23b919548049d21d4abbb8#file-xml2yolo-py ****************


def xml_to_yolo_bbox(bbox, w, h):
    # xmin, ymin, xmax, ymax
    x_center = ((bbox[2] + bbox[0]) / 2) / w
    y_center = ((bbox[3] + bbox[1]) / 2) / h
    width = (bbox[2] - bbox[0]) / w
    height = (bbox[3] - bbox[1]) / h
    return [x_center, y_center, width, height]


def yolo_to_xml_bbox(bbox, w, h):
    # x_center, y_center width heigth
    w_half_len = (bbox[2] * w) / 2
    h_half_len = (bbox[3] * h) / 2
    xmin = int((bbox[0] * w) - w_half_len)
    ymin = int((bbox[1] * h) - h_half_len)
    xmax = int((bbox[0] * w) + w_half_len)
    ymax = int((bbox[1] * h) + h_half_len)
    return [xmin, ymin, xmax, ymax]


def xml2yolo(input_dir, output_dir, image_dir):
    classes = []
    # create the labels folder (output directory)
    dirExists(output_dir)
    # identify all the xml files in the annotations folder (input directory)
    files = glob.glob(os.path.join(input_dir, "*.xml"))
    # loop through each
    for fil in files:
        basename = os.path.basename(fil)
        filename = os.path.splitext(basename)[0]
        # check if the label contains the corresponding image file
        if not os.path.exists(os.path.join(image_dir, f"{filename}.jpg")):
            print(f"{filename} image does not exist!")
            continue
        result = []
        # parse the content of the xml file
        tree = ET.parse(fil)
        root = tree.getroot()
        width = int(root.find("size").find("width").text)
        height = int(root.find("size").find("height").text)

        for obj in root.findall("object"):
            label = obj.find("name").text
            # check for new classes and append to list
            if label not in classes:
                classes.append(label)
            index = classes.index(label)
            pil_bbox = [int(x.text) for x in obj.find("bndbox")]
            yolo_bbox = xml_to_yolo_bbox(pil_bbox, width, height)
            # convert data to string
            bbox_string = " ".join([str(x) for x in yolo_bbox])
            result.append(f"{index} {bbox_string}")

        if result:
            # generate a YOLO format text file for each xml file
            with open(
                os.path.join(output_dir, f"{filename}.txt"), "w", encoding="utf-8"
            ) as f:
                f.write("\n".join(result))
    # generate the classes file as reference
    with open("classes.txt", "w", encoding="utf8") as f:
        f.write(json.dumps(classes))


# *********************** rearranging folders for yolov5 https://stackoverflow.com/questions/66238786/splitting-image-based-dataset-for-yolov3 *******************


def dirExists(name):
    if not os.path.isdir(name):
        os.mkdir(name)


def move(paths, folder):
    for p in paths:
        shutil.copy(p, folder)


def formatFolderStruct(f_loc, processed_folder, split_ratio):

    # Get all paths to your images files and text files

    PATH = f_loc + "/"
    img_paths = glob.glob(PATH + "images/*.jpg")
    txt_paths = glob.glob(PATH + "labels/*.txt")

    # Calculate number of files for training, validation

    data_size = len(img_paths)
    r = split_ratio
    train_size = int(data_size * r)

    # Now split them
    train_img_paths = img_paths[:train_size]
    train_txt_paths = txt_paths[:train_size]
    valid_img_paths = img_paths[train_size:]
    valid_txt_paths = txt_paths[train_size:]

    # Move them to train, valid folders
    dirExists("./yolov5/datasets")
    # newpath='datasets/fridgedata/'
    dirExists("./yolov5/datasets/fridgedata/")
    newpath_images = "./yolov5/datasets/fridgedata/images"
    dirExists(newpath_images)
    newpath_labels = "./yolov5/datasets/fridgedata/labels"
    dirExists(newpath_labels)

    # newpath='datasets/fridgedata'
    train_images = newpath_images + "/train/"
    valid_images = newpath_images + "/valid/"
    train_label = newpath_labels + "/train/"
    valid_label = newpath_labels + "/valid/"

    dirExists(train_images)
    dirExists(valid_images)
    dirExists(train_label)
    dirExists(valid_label)

    move(train_img_paths, train_images)
    move(train_txt_paths, train_label)
    move(valid_img_paths, valid_images)
    move(valid_txt_paths, valid_label)


# downloaddata(url, data_folder,data_file)
# print("data downloaded")

# xml2yolo(input_dir,output_dir,image_dir)
# print("formated to yolo")

formatFolderStruct(f_loc, processed_folder, split_ratio)
print("formated folder structure")
