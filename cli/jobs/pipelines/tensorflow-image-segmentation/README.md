# Tensorflow Distributed Image Segmentation

This folder provides a reference implementation for a tensorflow distributed training job. This implements an [image segmentation task based on a UNet architecture](https://keras.io/examples/vision/oxford_pets_image_segmentation/) on the [Oxford IIIT Pet dataset](https://www.robots.ox.ac.uk/~vgg/data/pets/).

The job uses the raw dataset and unpacks it as actual JPG/PNG files, instead of using [the `tfds` dataset](https://www.tensorflow.org/datasets/catalog/oxford_iiit_pet). The reason is that we want to provide you a job that you can easily transpose to your use case by changing the inputs files only.

Before submitting the pipeline, download the Oxford Pets archives from a machine
that can access the Oxford server, then upload them to your Azure ML workspace as
data assets:

```bash
curl -L https://www.robots.ox.ac.uk/~vgg/data/pets/data/images.tar.gz -o images.tar.gz
curl -L https://www.robots.ox.ac.uk/~vgg/data/pets/data/annotations.tar.gz -o annotations.tar.gz
az ml data create --name oxford-pets-images --version 1 --type uri_file --path ./images.tar.gz
az ml data create --name oxford-pets-annotations --version 1 --type uri_file --path ./annotations.tar.gz
```

Run these commands with Azure ML CLI configured for the target workspace (or
provide its resource group and workspace options). The `prepare_data` job reads
the uploaded assets in download mode and extracts them, so the compute cluster
does not need outbound access to `www.robots.ox.ac.uk`. If these data assets
already exist, update the input asset names or versions in `pipeline.yml` to
match your workspace.

We have tagged the code with the following expressions to walk you through the key implementation details.

Using your editor, search for those strings to get an idea of how to implement:
- DISTRIBUTED : how to implement distributed tensorflow,
- MLFLOW : how to implement mlflow reporting of metrics and artifacts,
- PROFILER: how to implement tensorflow profiling within a job.
