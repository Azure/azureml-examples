"""
This script provides code to load and setup a tensorflow segmentation model
"""
from tensorflow.keras import layers
from tensorflow import keras

MODEL_ARCH_MAP = {
    "unet": {"library": "keras"},
}

MODEL_ARCH_LIST = list(MODEL_ARCH_MAP.keys())


def get_model_metadata(model_arch: str):
    """Returns the model metadata"""
    if model_arch in MODEL_ARCH_MAP:
        return MODEL_ARCH_MAP[model_arch]
    else:
        raise NotImplementedError(f"model_arch={model_arch} is not implemented yet.")


def _get_unet_model(input_size, num_classes):
    """Constructs a UNET architecture model for segmentation"""
    # this code comes from https://keras.io/examples/vision/oxford_pets_image_segmentation/
    inputs = keras.Input(shape=input_size + (3,))

    ### [First half of the network: downsampling inputs] ###

    # Entry block
    x = layers.Conv2D(32, 3, strides=2, padding="same")(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    previous_block_activation = x  # Set aside residual

    # Blocks 1, 2, 3 are identical apart from the feature depth.
    for filters in [64, 128, 256]:
        x = layers.Activation("relu")(x)
        x = layers.SeparableConv2D(filters, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.Activation("relu")(x)
        x = layers.SeparableConv2D(filters, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.MaxPooling2D(3, strides=2, padding="same")(x)

        # Project residual
        residual = layers.Conv2D(filters, 1, strides=2, padding="same")(
            previous_block_activation
        )
        x = layers.add([x, residual])  # Add back residual
        previous_block_activation = x  # Set aside next residual

    ### [Second half of the network: upsampling inputs] ###

    for filters in [256, 128, 64, 32]:
        x = layers.Activation("relu")(x)
        x = layers.Conv2DTranspose(filters, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.Activation("relu")(x)
        x = layers.Conv2DTranspose(filters, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.UpSampling2D(2)(x)

        # Project residual
        residual = layers.UpSampling2D(2)(previous_block_activation)
        residual = layers.Conv2D(filters, 1, padding="same")(residual)
        x = layers.add([x, residual])  # Add back residual
        previous_block_activation = x  # Set aside next residual

    # Add a per-pixel classification layer
    outputs = layers.Conv2D(num_classes, 3, activation="softmax", padding="same")(x)

    # Define the model
    model = keras.Model(inputs, outputs)
    return model


def load_model(model_arch: str, input_size: int, num_classes: int):
    """Loads a model from a given arch and sets it up for training"""
    if model_arch not in MODEL_ARCH_MAP:
        raise NotImplementedError(f"model_arch={model_arch} is not implemented yet.")

    model = _get_unet_model((input_size, input_size), num_classes)

    return model
