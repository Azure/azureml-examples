# Converts MNIST-formatted files at the passed-in input path to training data output path and test data output path
import os


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


def prep(input_path, output_path_1, output_path_2):
    
    convert(
        os.path.join(input_path, "train-images-idx3-ubyte"),
        os.path.join(input_path, "train-labels-idx1-ubyte"),
        os.path.join(output_path_1, "mnist_train.csv"),
        60000,
    )
    convert(
        os.path.join(input_path, "t10k-images-idx3-ubyte"),
        os.path.join(input_path, "t10k-labels-idx1-ubyte"),
        os.path.join(output_path_2, "mnist_test.csv"),
        10000,
    )
