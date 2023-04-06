from base_jsonl_converter import JSONLConverter
import os
import xml.etree.ElementTree as ET

class VOCJSONLConverter(JSONLConverter):
    def __init__(self, base_url, xml_dir, segmentations_dir = None):
        super().__init__(base_url=base_url)
        self.xml_dir = xml_dir
        self.segmentations_dir = segmentations_dir 

    def convert(self):
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

            labels = []
            for object in root.findall("object"):
                name = object.find("name").text
                xmin = object.find("bndbox/xmin").text
                ymin = object.find("bndbox/ymin").text
                xmax = object.find("bndbox/xmax").text
                ymax = object.find("bndbox/ymax").text
                isCrowd = int(object.find("difficult").text)
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
            # build the jsonl file
            image_filename = root.find("filename").text
            _, file_extension = os.path.splitext(image_filename)
            json_line = dict(json_line_sample)
            json_line["image_url"] = os.path.join(json_line["image_url"],image_filename)
            json_line["image_details"]["format"] = file_extension[1:]
            json_line["image_details"]["width"] = width
            json_line["image_details"]["height"] = height
            json_line["label"] = labels

            self.jsonl_data.append(json_line)
        return self.jsonl_data