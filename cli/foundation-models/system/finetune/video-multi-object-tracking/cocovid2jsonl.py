import json
import os
import sys
import argparse

# Define Converters


class CocoVidToJSONLinesConverter:
    def convert(self):
        raise NotImplementedError


class BoundingBoxConverter(CocoVidToJSONLinesConverter):
    """example output for object tracking jsonl:
    {
      "image_url":"azureml://subscriptions/<my-subscription-id>/resourcegroups/<my-resource-group>/workspaces/<my-workspace>/datastores/<my-datastore>/paths/<path_to_image>",
      "image_details":{
          "format":"image_format",
          "width":"image_width",
          "height":"image_height"
      },
      "video_details": {
          "frame_id": "zero_based_frame_id(int)",
          "video_name": "video_name",
      },
      "label":[
          {
            "label":"class_name_1",
            "topX":"xmin/width",
            "topY":"ymin/height",
            "bottomX":"xmax/width",
            "bottomY":"ymax/height",
            "isCrowd":"isCrowd"
            "instance_id": "instance_id"
          },
          {
            "label":"class_name_2",
            "topX":"xmin/width",
            "topY":"ymin/height",
            "bottomX":"xmax/width",
            "bottomY":"ymax/height",
            "instance_id": "instance_id"
          },
          "..."
      ]
    }
    """

    def __init__(self, coco_data):
        self.json_lines_data = []
        self.categories = {}
        self.coco_data = coco_data
        self.image_id_to_data_index = {}
        self.video_id_to_name = {}
        for i in range(0, len(coco_data["images"])):
            self.json_lines_data.append({})
            self.json_lines_data[i]["image_url"] = ""
            self.json_lines_data[i]["image_details"] = {}
            self.json_lines_data[i]["video_details"] = {}
            self.json_lines_data[i]["label"] = []
        for i in range(0, len(coco_data["categories"])):
            self.categories[coco_data["categories"][i]["id"]] = coco_data["categories"][
                i
            ]["name"]
        for i in range(0, len(coco_data["videos"])):
            self.video_id_to_name[coco_data["videos"][i]["id"]] = coco_data["videos"][
                i
            ]["name"]

    def _populate_image_url(self, index, coco_image):
        self.json_lines_data[index]["image_url"] = coco_image["file_name"]
        self.image_id_to_data_index[coco_image["id"]] = index

    def _populate_image_details(self, index, coco_image):
        file_name = coco_image["file_name"]
        self.json_lines_data[index]["image_details"]["format"] = file_name[
            file_name.rfind(".") + 1 :
        ]
        self.json_lines_data[index]["image_details"]["width"] = coco_image["width"]
        self.json_lines_data[index]["image_details"]["height"] = coco_image["height"]

    def _populate_video_details(self, index, coco_image):
        self.json_lines_data[index]["video_details"]["frame_id"] = coco_image[
            "frame_id"
        ]
        self.json_lines_data[index]["video_details"][
            "video_name"
        ] = self.video_id_to_name[coco_image["video_id"]]

    def _populate_bbox_in_label(self, label, annotation, image_details):
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

    def _populate_label(self, annotation):
        index = self.image_id_to_data_index[annotation["image_id"]]
        image_details = self.json_lines_data[index]["image_details"]
        label = {"label": self.categories[annotation["category_id"]]}
        self._populate_bbox_in_label(label, annotation, image_details)
        self._populate_instanceId(label, annotation)
        self._populate_isCrowd(label, annotation)
        self._populate_visibility(label, annotation)
        self.json_lines_data[index]["label"].append(label)

    def _populate_instanceId(self, label, annotation):
        label["instance_id"] = annotation["instance_id"]

    def _populate_isCrowd(self, label, annotation):
        if "iscrowd" in annotation.keys():
            label["isCrowd"] = int(annotation["iscrowd"])

    def _populate_visibility(self, label, annotation):
        if "visibility" in annotation.keys():
            label["visibility"] = annotation["visibility"]

    def convert(self):
        for i in range(0, len(self.coco_data["images"])):
            self._populate_image_url(i, self.coco_data["images"][i])
            self._populate_image_details(i, self.coco_data["images"][i])
            self._populate_video_details(i, self.coco_data["images"][i])
        if "annotations" not in self.coco_data:
            self.coco_data["annotations"] = []
        for i in range(0, len(self.coco_data["annotations"])):
            self._populate_label(self.coco_data["annotations"][i])
        return self.json_lines_data


def main(args):
    input_coco_file_path = args.input_cocovid_file_path
    output_dir = args.output_dir
    output_file_path = output_dir + "/" + args.output_file_name
    print(output_file_path)
    task_type = args.task_type
    base_url = args.base_url

    def read_coco_file(coco_file):
        with open(coco_file) as f_in:
            return json.load(f_in)

    def write_json_lines(converter, filename, base_url=None):
        json_lines_data = converter.convert()
        with open(filename, "w") as outfile:
            for json_line in json_lines_data:
                if base_url is not None:
                    image_url = json_line["image_url"]
                    json_line["image_url"] = os.path.join(base_url, image_url)
                    json_line["image_url"] = json_line["image_url"].replace("\\", "/")
                json.dump(json_line, outfile, separators=(",", ":"))
                outfile.write("\n")
            print(f"Conversion completed. Converted {len(json_lines_data)} lines.")

    coco_data = read_coco_file(input_coco_file_path)

    print(f"Converting for {task_type}")

    if task_type == "ObjectTracking":
        converter = BoundingBoxConverter(coco_data)
        write_json_lines(converter, output_file_path, base_url)

    else:
        print("ERROR: Invalid Task Type")
        pass


if __name__ == "__main__":
    # Parse arguments that are passed into the script
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_cocovid_file_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--output_file_name", type=str, required=True)
    parser.add_argument(
        "--task_type",
        type=str,
        required=True,
        choices=["ObjectTracking"],
        default="ObjectTracking",
    )
    parser.add_argument("--base_url", type=str, default=None)

    args = parser.parse_args()
    main(args)
