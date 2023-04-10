from base_jsonl_converter import JSONLConverter
from azureml.automl.dnn.vision.object_detection.common import masktools
import pycocotools.mask as mask
import json

class COCOJSONLConverter(JSONLConverter):
    def __init__(self, base_url, coco_file):
        super().__init__(base_url=base_url)
        self.categories = {}
        self.coco_data = self.read_coco_file(coco_file)
        self.image_id_to_data_index = {}
        for i in range(0, len(self.coco_data["images"])):
            self.jsonl_data.append({})
            self.jsonl_data[i]["image_url"] = ""
            self.jsonl_data[i]["image_details"] = {}
            self.jsonl_data[i]["label"] = []
        for i in range(0, len(self.coco_data["categories"])):
            self.categories[self.coco_data["categories"][i]["id"]] = self.coco_data["categories"][
                i
            ]["name"]

    def read_coco_file(self, coco_file):
        with open(coco_file) as f_in:
            return json.load(f_in)

    def _populate_image_url(self, index, coco_image):
        # self.jsonl_data[index]["image_url"] = coco_image["file_name"]
        image_url = coco_image["file_name"]
        self.jsonl_data[index]["image_url"] = self.base_url + image_url[image_url.rfind("/") + 1 :]
        self.image_id_to_data_index[coco_image["id"]] = index

    def _populate_image_details(self, index, coco_image):
        file_name = coco_image["file_name"]
        self.jsonl_data[index]["image_details"]["format"] = file_name[
            file_name.rfind(".") + 1 :
        ]
        self.jsonl_data[index]["image_details"]["width"] = coco_image["width"]
        self.jsonl_data[index]["image_details"]["height"] = coco_image["height"]

    def _populate_bbox_in_label(self, label, annotation, image_details):
        if 'segmentation' not in annotation.keys() or len(annotation['segmentation']) == 0:
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
        else:
            label['bbox'] = 'null'

    def __populate_segmentation_in_label(self, label, annotation, image_details):
        # check if object detection or instance segmentation
        if 'segmentation' not in annotation.keys() or len(annotation['segmentation']) == 0:
            return

        # if bbox comes as normalized, skip normalization.
        if max(annotation["bbox"]) < 1.5:
            width = 1
            height = 1
        else:
            width = image_details["width"]
            height = image_details["height"]
        
        polygons = []
        if type(annotation['segmentation']) is dict: # segmentations are in uncompressed rle format
                rle = annotation['segmentation']
                compressed_rle = mask.frPyObjects(rle, rle['size'][0], rle['size'][1])
                polygons = masktools.convert_mask_to_polygon(compressed_rle)
        else: # segmentation is list of vertices
            for segmentation in annotation['segmentation']:
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

    def _populate_label(self, annotation):
        index = self.image_id_to_data_index[annotation["image_id"]]
        image_details = self.jsonl_data[index]["image_details"]
        label = {"label": self.categories[annotation["category_id"]]}
        self._populate_bbox_in_label(label, annotation, image_details)
        self.__populate_segmentation_in_label(label, annotation, image_details)
        self._populate_isCrowd(label, annotation)
        self.jsonl_data[index]["label"].append(label)

    def _populate_isCrowd(self, label, annotation):
        if "iscrowd" in annotation.keys():
            label["isCrowd"] = annotation["iscrowd"]

    def convert(self):
        for i in range(0, len(self.coco_data["images"])):
            self._populate_image_url(i, self.coco_data["images"][i])
            self._populate_image_details(i, self.coco_data["images"][i])
        for i in range(0, len(self.coco_data["annotations"])):
            self._populate_label(self.coco_data["annotations"][i])
        return self.jsonl_data