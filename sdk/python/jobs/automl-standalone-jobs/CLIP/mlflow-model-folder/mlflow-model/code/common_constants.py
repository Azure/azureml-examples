# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------

"""Mlflow PythonModel wrapper helper constants."""

from dataclasses import dataclass
from mlflow.types import DataType


@dataclass
class AugmentationConfigKeys:
    """Keys in augmentation configs"""

    TRAINING_PHASE_KEY = "train"
    VALIDATION_PHASE_KEY = "validation"
    AUGMENTATION_LIBRARY_NAME = "augmentation_library_name"
    OUTPUT_AUG_FILENAME = "augmentations.yaml"
    ALBUMENTATIONS = "albumentations"


@dataclass
class AlbumentationParameterNames:
    """ keys for Albumentations parameters"""
    TRANSFORMS_KEY = "transforms"
    BBOX_PARAMS = "bbox_params"
    PASCAL_VOC = "pascal_voc"
    CLASS_LABELS = "class_labels"
    IMAGE = "image"
    BBOX = "bbox"
    MASK = "mask"
    IMAGE_METADATA = "image_metadata"
    ORIGINAL_WIDTH = "original_width"
    ORIGINAL_HEIGHT = "original_height"
    NEW_WIDTH = "new_width"
    NEW_HEIGHT = "new_height"
    NEW_LEFT = "new_left"
    NEW_TOP = "new_top"
    WIDTH_SCALE = "width_scale"
    HEIGHT_SCALE = "height_scale"
    RESIZED_WIDTH = "resized_width"
    RESIZED_HEIGHT = "resized_height"


@dataclass
class AugmentationConfigFileExts:
    """Various Augmentation Config file types supported"""

    YAML = ".yaml"


class Tasks:
    "Tasks supported for All Frameworks"

    HF_MULTI_CLASS_IMAGE_CLASSIFICATION = "image-classification"
    HF_MULTI_LABEL_IMAGE_CLASSIFICATION = "image-classification-multilabel"
    MM_OBJECT_DETECTION = "image-object-detection"
    MM_INSTANCE_SEGMENTATION = "image-instance-segmentation"
    MM_MULTI_OBJECT_TRACKING = "video-multi-object-tracking"


class HFMiscellaneousLiterals:
    """HF miscellaneous constants"""

    PIXEL_VALUES = "pixel_values"
    DEFAULT_IMAGE_KEY = "image"
    IMAGE_FOLDER = "imagefolder"
    VAL = "val"
    ID2LABEL = "id2label"
    LABEL2ID = "label2id"


class HFConstants:
    """HF constants"""

    DEFAULT_DATALOADER_NUM_WORKERS = 6


class MLFlowSchemaLiterals:
    """MLFlow model signature related schema"""

    INPUT_IMAGE_KEY = "image_base64"
    INPUT_COLUMN_IMAGE_DATA_TYPE = DataType.binary
    INPUT_COLUMN_IMAGE = "image"
    OUTPUT_COLUMN_DATA_TYPE = DataType.string
    OUTPUT_COLUMN_FILENAME = "filename"
    OUTPUT_COLUMN_PROBS = "probs"
    OUTPUT_COLUMN_LABELS = "labels"
    OUTPUT_COLUMN_BOXES = "boxes"

    BATCH_SIZE_KEY = "batch_size"
    SCHEMA_SIGNATURE = "signature"
    TRAIN_LABEL_LIST = "train_label_list"
    WRAPPER = "images_model_wrapper"
    THRESHOLD = "threshold"


class MMDetLiterals:
    """MMDetection constants"""
    CONFIG_PATH = "config_path"
    WEIGHTS_PATH = "weights_path"
    AUGMENTATIONS_PATH = "augmentations_path"
    METAFILE_PATH = "model_metadata"
    MODEL_DEFAULTS_PATH = "model_defaults_path"


class MmDetectionDatasetLiterals:
    """MMDetection dataset constants"""

    IMG = "img"
    IMG_METAS = "img_metas"
    GT_BBOXES = "gt_bboxes"
    GT_LABELS = "gt_labels"
    GT_CROWDS = "gt_crowds"
    GT_MASKS = "gt_masks"
    MASKS = "masks"
    BBOXES = "bboxes"
    LABELS = "labels"
    IMAGE_SHAPE = "img_shape"
    IMAGE_ORIGINAL_SHAPE = "ori_shape"
    RAW_DIMENSIONS = "raw_dimensions"
    RAW_MASK_DIMENSIONS = "raw_mask_dimensions"


class ODLiterals:
    """OD constants"""

    LABEL = "label"
    BOXES = "boxes"
    SCORE = "score"
    BOX = "box"
    TOP_X = "topX"
    TOP_Y = "topY"
    BOTTOM_X = "bottomX"
    BOTTOM_Y = "bottomY"
    POLYGON = "polygon"


class MmDetectionConfigLiterals:
    """MMDetection config constants"""

    NUM_CLASSES = "num_classes"
    BOX_SCORE_THRESHOLD = "score_thr"


class MetricsLiterals:
    """ Azureml Metrics Literals"""

    SCORES = "scores"
    CLASSES = "classes"
    METRICS_COMPUTER = "metrics_computer"
    METRICS = "metrics"
