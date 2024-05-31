# Tensorflow Distributed Image Segmentation

This folder provides a reference implementation for a tensorflow distributed training job. This implements an [image segmentation task based on a UNet architecture](https://keras.io/examples/vision/oxford_pets_image_segmentation/) on the [Oxford IIIT Pet dataset](https://www.robots.ox.ac.uk/~vgg/data/pets/).

The job uses the raw dataset and unpacks it as actual JPG/PNG files, instead of using [the `tfds` dataset](https://www.tensorflow.org/datasets/catalog/oxford_iiit_pet). The reason is that we want to provide you a job that you can easily transpose to your use case by changing the inputs files only.

We have tagged the code with the following expressions to walk you through the key implementation details.

Using your editor, search for those strings to get an idea of how to implement:
- DISTRIBUTED : how to implement distributed tensorflow,
- MLFLOW : how to implement mlflow reporting of metrics and artifacts,
- PROFILER: how to implement tensorflow profiling within a job.
