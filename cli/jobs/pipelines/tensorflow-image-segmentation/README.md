# Tensorflow Distributed Image Segmentation

This folder provides a reference implementation for a tensorflow distributed training job. This implements an [image segmentation task based on a UNet architecture](https://keras.io/examples/vision/oxford_pets_image_segmentation/) on the [Oxford IIIT Pet dataset](https://www.robots.ox.ac.uk/~vgg/data/pets/).

The job uses the raw dataset and unpacks it as actual JPG/PNG files, instead of using [the `tfds` dataset](https://www.tensorflow.org/datasets/catalog/oxford_iiit_pet). The reason is that we want to provide you a job that you can easily transpose to your use case by changing the inputs files only.

Before submitting the pipeline, run the preparation script from this directory
on a machine that can access the Oxford server:

```bash
bash prepare-data.sh
az ml job create --file pipeline.yml
```

Configure Azure ML CLI defaults for the target resource group and workspace
before running these commands. The script downloads and validates both archives,
then uploads them as `oxford-pets-images:1` and `oxford-pets-annotations:1` data
assets. The GitHub Actions workflow runs this script before submitting the job;
it stops if downloading, validation, or registration fails.

The `prepare_data` job reads the uploaded assets in download mode and extracts
them, so the compute cluster does not need outbound access to
`www.robots.ox.ac.uk`. The preparation machine still needs access to that site.
If you see `Could not resolve uris of type data`, check that both assets exist
in the workspace where you submit the pipeline. Registering a public Oxford URL
as an asset does not upload its contents; use locally downloaded files as the
script does. For existing data under different names or versions, update the
input paths in `pipeline.yml` to match.

We have tagged the code with the following expressions to walk you through the key implementation details.

Using your editor, search for those strings to get an idea of how to implement:
- DISTRIBUTED : how to implement distributed tensorflow,
- MLFLOW : how to implement mlflow reporting of metrics and artifacts,
- PROFILER: how to implement tensorflow profiling within a job.
