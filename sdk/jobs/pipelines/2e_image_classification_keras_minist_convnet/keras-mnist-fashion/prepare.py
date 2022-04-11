import os
import sys
import argparse

parser = argparse.ArgumentParser("prepare")
parser.add_argument("--mounted_input_path", type=str, help="Input path of original data")
parser.add_argument("--mounted_output_path", type=str, help="Output path of converted data")

args = parser.parse_args()

def convert(imgf, labelf, outf, n):
    f = open(imgf, "rb")
    l = open(labelf, "rb")
    o = open(outf, "w")

    f.read(16)
    l.read(8)
    images = []

    for i in range(n):
        image = [ord(l.read(1))]
        for j in range(28 * 28):
            image.append(ord(f.read(1)))
        images.append(image)

    for image in images:
        o.write(",".join(str(pix) for pix in image) + "\n")
    f.close()
    o.close()
    l.close()


mounted_input_path = args.mounted_input_path
mounted_output_path = args.mounted_output_path
os.makedirs(mounted_output_path, exist_ok=True)
print("mounted_input_path")
print(mounted_input_path)

print("mounted_path files: ")
arr = os.listdir(mounted_input_path)
print(arr)

convert(
    os.path.join(mounted_input_path, "train-images-idx3-ubyte"),
    os.path.join(mounted_input_path, "train-labels-idx1-ubyte"),
    os.path.join(mounted_output_path, "mnist_train.csv"),
    60000,
)
convert(
    os.path.join(mounted_input_path, "t10k-images-idx3-ubyte"),
    os.path.join(mounted_input_path, "t10k-labels-idx1-ubyte"),
    os.path.join(mounted_output_path, "mnist_test.csv"),
    10000,
)
