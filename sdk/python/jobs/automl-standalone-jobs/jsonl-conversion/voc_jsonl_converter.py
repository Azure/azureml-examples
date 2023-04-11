from base_jsonl_converter import JSONLConverter
import os
import xml.etree.ElementTree as ET
import numpy as np
import PIL.Image as Image
from azureml.automl.dnn.vision.object_detection.common import masktools
import torch


class VOCJSONLConverter(JSONLConverter):
    """
    Class for converting VOC data for object detection and instance segmentation into jsonl files
    ...
    Attributes
    ---------
    base_url : str
        the base for the image_url to be written into the jsonl file
    xml_dir : str
        directory of xml annotation files
    mask_dir : str
        directory of png masks for instance segmentation
    """

    def __init__(self, base_url, xml_dir, mask_dir=None):
        super().__init__(base_url=base_url)
        self.xml_dir = xml_dir
        self.mask_dir = mask_dir

    def convert(self):
        """
        Generate jsonl data for object detection or instance segmentation

        return: list of lines for jsonl
        rtype: List <class 'dict'>

        """
        json_line_sample = {
            "image_url": self.base_url,
            "image_details": {"format": None, "width": None, "height": None},
            "label": [],
        }

        for i, filename in enumerate(os.listdir(self.xml_dir)):
            if not filename.endswith(".xml"):
                print(f"Skipping unknown file: {filename}")
                continue

            annotation_filename = os.path.join(self.xml_dir, filename)
            print(f"Parsing {annotation_filename}")

            root = ET.parse(annotation_filename).getroot()
            width = int(root.find("size/width").text)
            height = int(root.find("size/height").text)

            # convert mask into polygon
            if self.mask_dir is not None:
                mask_fname = os.path.join(self.mask_dir, filename[:-4] + ".png")
                polygons = self.parsing_mask(mask_fname)

            labels = []
            for index, object in enumerate(root.findall("object")):
                name = object.find("name").text
                isCrowd = int(object.find("difficult").text)
                if self.mask_dir is None:  # do object detection
                    xmin = object.find("bndbox/xmin").text
                    ymin = object.find("bndbox/ymin").text
                    xmax = object.find("bndbox/xmax").text
                    ymax = object.find("bndbox/ymax").text

                    labels.append(
                        {
                            "label": name,
                            "topX": float(xmin) / width,
                            "topY": float(ymin) / height,
                            "bottomX": float(xmax) / width,
                            "bottomY": float(ymax) / height,
                            "isCrowd": isCrowd,
                        }
                    )
                else:
                    labels.append(
                        {
                            "label": name,
                            "bbox": "null",
                            "isCrowd": isCrowd,
                            "polygon": polygons[index],
                        }
                    )
            # build the jsonl file
            image_filename = root.find("filename").text
            _, file_extension = os.path.splitext(image_filename)
            json_line = dict(json_line_sample)
            json_line["image_url"] = os.path.join(
                json_line["image_url"], image_filename
            )
            json_line["image_details"]["format"] = file_extension[1:]
            json_line["image_details"]["width"] = width
            json_line["image_details"]["height"] = height
            json_line["label"] = labels

            self.jsonl_data.append(json_line)
        return self.jsonl_data

    def parsing_mask(self, mask_fname):
        """
        Parse mask from png in the same order as annotations in the xml file

        Parameters:
            mask_fname (str): name of mask png to parse

        return: list of polygons
        rtype: List
        """
        binary_masks = self.binarise_mask(mask_fname)
        polygons = []
        for bi_mask in binary_masks:

            if len(bi_mask.shape) == 2:
                bi_mask = bi_mask[np.newaxis, :]
            polygon = self.convert_mask_to_polygon(bi_mask)
            polygons.append(polygon)

        return polygons

    def binarise_mask(self, mask_fname):
        """
        Binarize mask

        Parameters:
            mask_fname (str): name of mask png to parse

        return: list of polygons
        rtype: List
        """

        mask = Image.open(mask_fname)
        mask = np.array(mask)
        # instances are encoded as different colors
        obj_ids = np.unique(mask)
        # first id is the background, so remove it
        obj_ids = obj_ids[1:]

        # split the color-encoded mask into a set of binary masks
        binary_masks = mask == obj_ids[:, None, None]
        return binary_masks

    def convert_mask_to_polygon(
        self,
        mask,
        score_threshold=0.5,
    ):
        """Convert a numpy mask to a polygon outline in normalized coordinates.

        :param mask: Pixel mask, where each pixel has an object (float) score in [0, 1], in size ([1, height, width])
        :type: mask: <class 'numpy.array'>
        :param score_threshold: Score cutoff for considering a pixel as in object.
        :type: score_threshold: Float
        :return: normalized polygon coordinates
        :rtype: list of list
        """
        # Convert to numpy bitmask
        mask = mask[0]
        mask_array = np.array((mask > score_threshold), dtype=np.uint8)

        rle_mask = masktools.encode_mask_as_rle(torch.from_numpy(mask_array))
        return masktools.convert_mask_to_polygon(rle_mask)
