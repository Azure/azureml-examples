# Fine-tuning a model for Image Instance Segmentation task

You can launch a sample pipeline for image instance segmentation using `mmdetection_image_objectdetection_instancesegmentation_pipeline` component.

For using this component for instance segmentation, run the shell script file `bash ./mmdetection-fridgeobjects-instance-segmentation.sh`.

Currently following models are supported:
| Model Name | Source |
| :------------: | :-------:  |
| [mask_rcnn_swin-t-p4-w7_fpn_1x_coco](https://ml.azure.com/registries/azureml-staging/models/mask_rcnn_swin-t-p4-w7_fpn_1x_coco/version/1?tid=72f988bf-86f1-41af-91ab-2d7cd011db47#overview) | azureml-staging registry |
| [Image instance-segmentation models from MMDetection](https://github.com/open-mmlab/mmdetection/blob/v2.28.2/docs/en/model_zoo.md) | MMDetection |
