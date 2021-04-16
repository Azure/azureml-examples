import sys
import argparse
import mlflow.fastai
import fastai.vision as vis


def main():
    # Parse command-line arguments
    epochs = int(sys.argv[1]) if len(sys.argv) > 2 else 5
    lr = float(sys.argv[2]) if len(sys.argv) > 1 else 0.01

    # Download and untar the MNIST data set
    path = vis.untar_data(vis.URLs.MNIST_TINY)

    # Prepare, transform, and normalize the data
    data = vis.ImageDataBunch.from_folder(
        path, ds_tfms=(vis.rand_pad(2, 28), []), bs=64
    )
    data.normalize(vis.imagenet_stats)

    # Train and fit the Learner model
    learn = vis.cnn_learner(data, vis.models.resnet18, metrics=vis.accuracy)

    # Enable auto logging
    mlflow.fastai.autolog()

    # Train and fit with default or supplied command line arguments
    learn.fit(epochs, lr)


if __name__ == "__main__":
    main()
