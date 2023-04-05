import os
from base_jsonl_converter import JSONLConverter

class ClassificationJSONLConverter(JSONLConverter):

    def __init__(self, base_url, data_dir = None, label_file=None):
        self.label_file = label_file
        self.data_dir = data_dir
        super().__init__(base_url)

    def convert(self):
        if self.label_file is not None: # multilabel classification
            return self.multilabel2jsonl()
        elif self.data_dir is not None: # multiclass classification
            return self.multiclass2jsonl()
        else:
            return None 


    def multiclass2jsonl(self):
        # Baseline of json line dictionary
        json_line_sample = {
            "image_url": self.base_url,
            "label": "",
        }
        index = 0
        # Scan each sub directary and generate a jsonl line per image, distributed on train and valid JSONL files
        for class_name in os.listdir(self.data_dir):
            sub_dir = os.path.join(self.data_dir, class_name)
            if not os.path.isdir(sub_dir):
                continue

            # Scan each sub directary
            print(f"Parsing {sub_dir}")
            for image in os.listdir(sub_dir):
                json_line = dict(json_line_sample)
                json_line["image_url"] += f"{class_name}/{image}"
                json_line["label"] = class_name

                self.jsonl_data.append(json_line)
                index += 1

        return self.jsonl_data

    def multilabel2jsonl(self):
        # Baseline of json line dictionary
        json_line_sample = {
            "image_url": self.base_url,
            "label": [],
        }

        # Read each annotation and convert it to jsonl line
        with open(self.label_file, "r") as labels:
            for i, line in enumerate(labels):
                # Skipping the title line and any empty lines.
                if i == 0 or len(line.strip()) == 0:
                    continue
                line_split = line.strip().split(",")
                if len(line_split) != 2:
                    print(f"Skipping the invalid line: {line}")
                    continue
                json_line = dict(json_line_sample)
                json_line["image_url"] += f"images/{line_split[0]}"
                json_line["label"] = line_split[1].strip().split(" ")

                self.jsonl_data.append(json_line)

        return self.jsonl_data
