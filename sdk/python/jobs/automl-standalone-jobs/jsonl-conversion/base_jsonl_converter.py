import json

# ------Start Class ------#
class JSONLConverter:
    """
    Base class for JSONL converters
    ...
    Attributes
    ---------
    base_url : str
        the base for the image_url to be written into the jsonl file
    """

    def __init__(self, base_url):
        self.jsonl_data = []
        self.base_url = base_url

    def convert(self):
        raise NotImplementedError


# ------End Class------#


def write_json_lines(converter, filename):
    """
    Converts and writes a jsonl file

    Parameters:
        converter (JSONLConverter): the converter use to generate the jsonl
        filename (str): output file for writing jsonl
        base_url (str): the base for the image_url to be written into the jsonl file
    """
    json_lines_data = converter.convert()
    with open(filename, "w") as outfile:
        for json_line in json_lines_data:
            json.dump(json_line, outfile, separators=(",", ":"))
            outfile.write("\n")
        print(f"Conversion completed. Converted {len(json_lines_data)} lines.")
