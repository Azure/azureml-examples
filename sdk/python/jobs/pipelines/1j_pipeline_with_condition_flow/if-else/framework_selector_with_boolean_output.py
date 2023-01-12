from mldesigner import Output, command_component

class Tasks:
    """Defines types of machine learning tasks supported by automated ML."""

    IMAGE_CLASSIFICATION = 'image-classification'
    IMAGE_CLASSIFICATION_MULTILABEL = 'image-classification-multilabel'
    IMAGE_OBJECT_DETECTION = 'image-object-detection'
    IMAGE_INSTANCE_SEGMENTATION = 'image-instance-segmentation'


def image_classification_framework_selector(model_name: str):
    image_classification_models_runtime = ['mobilenetv2',
                                           'resnet18',
                                           'resnet34',
                                           'resnet50',
                                           'resnet101',
                                           'resnet152',
                                           'resnest50',
                                           'resnest101',
                                           'seresnext',
                                           'vits16r224',
                                           'vitb16r224',
                                           'vitl16r224']

    if (model_name in image_classification_models_runtime):
        return True

    return False


def image_object_detection_framework_selector(model_name: str):
    image_object_detection_models_runtime = ['yolov5',
                                             'fasterrcnn_resnet18_fpn',
                                             'fasterrcnn_resnet34_fpn',
                                             'fasterrcnn_resnet50_fpn',
                                             'fasterrcnn_resnet101_fpn',
                                             'fasterrcnn_resnet152_fpn',
                                             'retinanet_resnet50_fpn']

    if (model_name in image_object_detection_models_runtime):
        return True

    return False


def image_instance_segmentation_framework_selector(model_name: str):
    image_instance_segmentation_models_runtime = ['maskrcnn_resnet18_fpn',
                                                  'maskrcnn_resnet34_fpn',
                                                  'maskrcnn_resnet50_fpn',
                                                  'maskrcnn_resnet101_fpn',
                                                  'maskrcnn_resnet152_fpn']

    if (model_name in image_instance_segmentation_models_runtime):
        return True

    return False

@command_component()
def framework_selector() -> Output(type="boolean", is_control=True):
    task = 'image-object-detection' #task_type
    model = 'yolov5' #model_name

    if task == Tasks.IMAGE_CLASSIFICATION or task == Tasks.IMAGE_CLASSIFICATION_MULTILABEL:
        return image_classification_framework_selector(model)

    elif task == Tasks.IMAGE_OBJECT_DETECTION:
        return image_object_detection_framework_selector(model)

    elif task == Tasks.IMAGE_INSTANCE_SEGMENTATION:
        return image_instance_segmentation_framework_selector(model)

    else:
        return True
    
    return True
