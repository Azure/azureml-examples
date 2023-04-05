import json

#------Start Class ------#
class JSONLConverter():

    def __init__(self, base_url):
        self.jsonl_data = []
        self.base_url = base_url
    
    def convert(self):
        raise NotImplementedError
    
#------End Class------#


def write_json_lines(converter, filename, base_url=None):
    json_lines_data = converter.convert()
    with open(filename, "w") as outfile:
        for json_line in json_lines_data:
            if base_url is not None:
                image_url = json_line["image_url"]
                json_line["image_url"] = (
                    base_url + image_url[image_url.rfind("/") + 1 :]
                )
            json.dump(json_line, outfile, separators=(",", ":"))
            outfile.write("\n")
        print(f"Conversion completed. Converted {len(json_lines_data)} lines.")