# Fine-tuning a model for Image Object Detection task

You can launch a sample pipeline for image object detection using `mmdetection_image_objectdetection_instancesegmentation_pipeline` component.

For using this component for object detection, run the shell script file `bash ./mmdetection-fridgeobjects-detection.sh`.

Currently following models are supported:
| Model Name | Source |
| :------------: | :-------:  |
| [deformable_detr_twostage_refine_r50_16x2_50e_coco](https://ml.azure.com/registries/azureml/models/deformable_detr_twostage_refine_r50_16x2_50e_coco/version/3) | azureml registry |
| [sparse_rcnn_r50_fpn_300_proposals_crop_mstrain_480-800_3x_coco](https://ml.azure.com/registries/azureml/models/sparse_rcnn_r50_fpn_300_proposals_crop_mstrain_480-800_3x_coco/version/3) | azureml registry |
| [sparse_rcnn_r101_fpn_300_proposals_crop_mstrain_480-800_3x_coco](https://ml.azure.com/registries/azureml/models/sparse_rcnn_r101_fpn_300_proposals_crop_mstrain_480-800_3x_coco/version/3) | azureml registry |
| [vfnet_r50_fpn_mdconv_c3-c5_mstrain_2x_coco](https://ml.azure.com/registries/azureml/models/vfnet_r50_fpn_mdconv_c3-c5_mstrain_2x_coco/version/3) | azureml registry |
| [vfnet_x101_64x4d_fpn_mdconv_c3-c5_mstrain_2x_coco](https://ml.azure.com/registries/azureml/models/vfnet_x101_64x4d_fpn_mdconv_c3-c5_mstrain_2x_coco/version/3) | azureml registry |
| [yolof_r50_c5_8x8_1x_coco](https://ml.azure.com/registries/azureml/models/yolof_r50_c5_8x8_1x_coco/version/3) | azureml registry |
| [Image object detection models from MMDetection](https://github.com/open-mmlab/mmdetection/blob/v2.28.2/docs/en/model_zoo.md) | MMDetection |
