# Fine-tuning a model for Image Object Detection task

You can launch a sample pipeline for image object detection using `mmdetection_image_objectdetection_instancesegmentation_pipeline` component.

For using this component for object detection, run the shell script file `bash ./mmdetection-fridgeobjects-detection.sh`.

Currently following models are supported:

| Model Name | Source |
| :------------: | :-------:  |
| [mmd-3x-deformable-detr_refine_twostage_r50_16xb2-50e_coco](https://ml.azure.com/registries/azureml/models/mmd-3x-deformable-detr_refine_twostage_r50_16xb2-50e_coco/version/12) | azureml registry |
| [mmd-3x-sparse-rcnn_r50_fpn_300-proposals_crop-ms-480-800-3x_coco](https://ml.azure.com/registries/azureml/models/mmd-3x-sparse-rcnn_r50_fpn_300-proposals_crop-ms-480-800-3x_coco/version/12) | azureml registry |
| [mmd-3x-sparse-rcnn_r101_fpn_300-proposals_crop-ms-480-800-3x_coco](https://ml.azure.com/registries/azureml/models/mmd-3x-sparse-rcnn_r101_fpn_300-proposals_crop-ms-480-800-3x_coco/version/12) | azureml registry |
| [mmd-3x-vfnet_r50-mdconv-c3-c5_fpn_ms-2x_coco](https://ml.azure.com/registries/azureml/models/mmd-3x-vfnet_r50-mdconv-c3-c5_fpn_ms-2x_coco/version/12) | azureml registry |
| [mmd-3x-vfnet_x101-64x4d-mdconv-c3-c5_fpn_ms-2x_coco](https://ml.azure.com/registries/azureml/models/mmd-3x-vfnet_x101-64x4d-mdconv-c3-c5_fpn_ms-2x_coco/version/12) | azureml registry |
| [mmd-3x-yolof_r50_c5_8x8_1x_coco](https://ml.azure.com/registries/azureml/models/mmd-3x-yolof_r50_c5_8x8_1x_coco/version/12) | azureml registry |
| [Image object detection models from MMDetection](https://github.com/open-mmlab/mmdetection/blob/v3.1.0/docs/en/model_zoo.md) | MMDetection |
