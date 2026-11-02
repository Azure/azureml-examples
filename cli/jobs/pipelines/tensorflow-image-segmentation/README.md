# Tensorflow Distributed Image Segmentation

This folder provides a reference implementation for a tensorflow distributed training job. This implements an [image segmentation task based on a UNet architecture](https://keras.io/examples/vision/oxford_pets_image_segmentation/) on the [Oxford IIIT Pet dataset](https://www.robots.ox.ac.uk/~vgg/data/pets/).

The job uses the raw dataset and unpacks it as actual JPG/PNG files, instead of using [the `tfds` dataset](https://www.tensorflow.org/datasets/catalog/oxford_iiit_pet). The reason is that we want to provide you a job that you can easily transpose to your use case by changing the inputs files only.

The `dataset_archive` input uses [FastAI's Oxford-IIIT Pet archive](https://s3.amazonaws.com/fast-ai-imageclas/oxford-iiit-pet.tgz), which contains both images and annotations, instead of downloading from the Oxford hosts that can time out. It uses `mode: download` so Azure ML downloads the complete file before the `prepare_data` command extracts it, rather than mounting an external HTTPS URL. Extraction removes the archive's top-level `oxford-iiit-pet/` directory, preserving the `images/` and `annotations/trimaps/` paths used by training.

The compute must be able to reach `s3.amazonaws.com` over HTTPS. For restricted networks, upload this combined archive to your workspace datastore and replace `jobs.prepare_data.inputs.dataset_archive.path` in `pipeline.yml` with its datastore URI. You can also set that path to a local copy when submitting with `az ml job create`; Azure ML uploads local inputs to workspace storage. The original, separate Oxford archives are not interchangeable with this combined input.

We have tagged the code with the following expressions to walk you through the key implementation details.

Using your editor, search for those strings to get an idea of how to implement:
- DISTRIBUTED : how to implement distributed tensorflow,
- MLFLOW : how to implement mlflow reporting of metrics and artifacts,
- PROFILER: how to implement tensorflow profiling within a job.
