import os
import argparse
import gzip
import idx2numpy
import urllib.request
from PIL import Image

parser = argparse.ArgumentParser(allow_abbrev=False, description="parse user arguments")
parser.add_argument("--output_folder", type=str, default=0)

args, _ = parser.parse_known_args()

data_folder = os.path.join(args.output_folder, "mnist")
os.makedirs(data_folder, exist_ok=True)
urllib.request.urlretrieve(
    "https://azureopendatastorage.blob.core.windows.net/mnist/t10k-images-idx3-ubyte.gz",
    filename=os.path.join(os.getcwd(), "test-images.gz"),
)

file_handler = gzip.open("test-images.gz", "r")
imagearray = idx2numpy.convert_from_file(file_handler)

# Choose the first 1000 images and save to the output folder.
for i in range(1000):
    im = Image.fromarray(imagearray[i])
    im.save(os.path.join(data_folder, f"{i}.png"))

print("Saved 1000 images to the output folder")
