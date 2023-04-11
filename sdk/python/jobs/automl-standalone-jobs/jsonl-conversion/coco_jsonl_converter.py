from base_jsonl_converter import JSONLConverter
from azureml.automl.dnn.vision.object_detection.common import masktools
import pycocotools.mask as mask
import json


class COCOJSONLConverter(JSONLConverter):
    """
    Class for converting COCO data for object detection and instance segmentation into jsonl files
    ...
    Attributes
    ---------
    base_url : str
        the base for the image_url to be written into the jsonl file
    coco_file : str
        file containing coco annotations
    compressed_rle : bool
        flag indicating if coco segmentation annotations are stored in comprssed rle format
    """

    def __init__(self, base_url, coco_file, compressed_rle=False):
        super().__init__(base_url=base_url)
        self.categories = {}
        self.compressed_rle = compressed_rle
        with open(coco_file) as f_in:
            self.coco_data = json.load(f_in)
        self.image_id_to_data_index = {}
        for i in range(0, len(self.coco_data["images"])):
            self.jsonl_data.append({})
            self.jsonl_data[i]["image_url"] = ""
            self.jsonl_data[i]["image_details"] = {}
            self.jsonl_data[i]["label"] = []
        for i in range(0, len(self.coco_data["categories"])):
            self.categories[self.coco_data["categories"][i]["id"]] = self.coco_data[
                "categories"
            ][i]["name"]

    def convert(self):
        """
        Generate jsonl data for object detection or instance segmentation

        return: list of lines for jsonl
        rtype: List <class 'dict'>

        """
        for i in range(0, len(self.coco_data["images"])):
            self._populate_image_url(i, self.coco_data["images"][i])
            self._populate_image_details(i, self.coco_data["images"][i])
        for i in range(0, len(self.coco_data["annotations"])):
            self._populate_label(self.coco_data["annotations"][i])
        return self.jsonl_data

    def _populate_image_url(self, index, coco_image):
        """
        populates image url for jsonl entry

        Parameters:
            index (int): image entry index
            coco_image (dict): image entry from coco data file
        """
        image_url = coco_image["file_name"]
        self.jsonl_data[index]["image_url"] = (
            self.base_url + image_url[image_url.rfind("/") + 1 :]
        )
        self.image_id_to_data_index[coco_image["id"]] = index

    def _populate_image_details(self, index, coco_image):
        """
        populates image details for jsonl entry

        Parameters:
            index (int): image entry index
            coco_image (dict): image entry from coco data file
        return: list of lines for jsonl
        """
        file_name = coco_image["file_name"]
        self.jsonl_data[index]["image_details"]["format"] = file_name[
            file_name.rfind(".") + 1 :
        ]
        self.jsonl_data[index]["image_details"]["width"] = coco_image["width"]
        self.jsonl_data[index]["image_details"]["height"] = coco_image["height"]

    def _populate_label(self, annotation):
        """
        populates label entry for object detection or instance segmentation

        Parameters:
            annotation (dict): annotation entry from coco data file
        """
        index = self.image_id_to_data_index[annotation["image_id"]]
        image_details = self.jsonl_data[index]["image_details"]
        label = {"label": self.categories[annotation["category_id"]]}
        # check if object detection or instance segmentation
        if (
            "segmentation" not in annotation.keys()
            or len(annotation["segmentation"]) == 0
        ):
            self._populate_bbox_in_label(label, annotation, image_details)
        else:
            self.__populate_segmentation_in_label(label, annotation, image_details)
        self._populate_isCrowd(label, annotation)
        self.jsonl_data[index]["label"].append(label)

    def _populate_bbox_in_label(self, label, annotation, image_details):
        """
        populates bounding box in label entry for object detection

        Parameters:
            label (dict): label to populate for jsonl entry
            annotation (dict): annotation entry from coco data file
        """
        # if bbox comes as normalized, skip normalization.
        if max(annotation["bbox"]) < 1.5:
            width = 1
            height = 1
        else:
            width = image_details["width"]
            height = image_details["height"]
        label["topX"] = annotation["bbox"][0] / width
        label["topY"] = annotation["bbox"][1] / height
        label["bottomX"] = (annotation["bbox"][0] + annotation["bbox"][2]) / width
        label["bottomY"] = (annotation["bbox"][1] + annotation["bbox"][3]) / height

    def __populate_segmentation_in_label(self, label, annotation, image_details):
        """
        populates polygon segmentation in label entry for instance segmentation

        Parameters:
            label (dict): label to populate for jsonl entry
            annotation (dict): annotation entry from coco data file
            image_details (dict): image details from coco data file
        """
        # if bbox comes as normalized, skip normalization.
        if max(annotation["bbox"]) < 1.5:
            width = 1
            height = 1
        else:
            width = image_details["width"]
            height = image_details["height"]

        polygons = []
        if (
            type(annotation["segmentation"]) is dict
        ):  # segmentations are in uncompressed rle format
            rle = annotation["segmentation"]
            if self.compressed_rle:
                compressed_rle = rle
            else:
                compressed_rle = mask.frPyObjects(rle, rle["size"][0], rle["size"][1])
            polygons = masktools.convert_mask_to_polygon(compressed_rle)
        else:  # segmentation is list of vertices
            for segmentation in annotation["segmentation"]:
                polygon = []
                # loop through vertices:
                for id, vertex in enumerate(segmentation):
                    if (id % 2) == 0:
                        # x-coordinates (even index)
                        x = vertex / width
                        polygon.append(x)

                    else:
                        y = vertex / height
                        polygon.append(y)
                polygons.append(polygon)
        label["polygon"] = polygons

    def _populate_isCrowd(self, label, annotation):
        """
        populates iscrowd in label entry for object detection and instance segmentation

        Parameters:
            label (dict): label to populate for json entry
            annotation (dict): annotation entry from coco data file
        """
        if "iscrowd" in annotation.keys():
            label["isCrowd"] = annotation["iscrowd"]
