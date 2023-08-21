# MMDetection Model Import Component
The component copies the input model folder to the component output directory when the model is passed as an input to the `pytorch_model` or `mlflow_model` nodes. If `model_name `is selected, the component will download the config for the model from [MMDetection model zoo](https://github.com/open-mmlab/mmdetection/blob/v2.28.2/docs/en/model_zoo.md). The component can be seen in your workspace component page - [mmdetection_image_objectdetection_instancesegmentation_model_import](https://ml.azure.com/registries/azureml/components/mmdetection_image_objectdetection_instancesegmentation_model_import).

# 1. Inputs

1. _pytorch_model_ (custom_model, optional)

    Pytorch Model registered in AzureML Asset.

    The continual finetune flag will be set to true in this case. If you want to resume from previous training state, set *resume_from_checkpoint* flag to True in [finetune component](mmd_finetune_component.md/#39-resume-from-checkpoint).

2. _mlflow_model_ (mlflow_model, optional)

    MlFlow Model registered in AzureML Asset. Some MMDetection models are registered in azureml registry and can be used directly. The user can also register MMDetection models into their workspace or organisation's registry, and use them.

    Following models are registered in azureml registry, and can be used directly.
    | Model Name | Source |
    | :------------: | :-------:  |
    | [deformable_detr_twostage_refine_r50_16x2_50e_coco](https://ml.azure.com/registries/azureml/models/deformable_detr_twostage_refine_r50_16x2_50e_coco/version/3) | azureml registry |
    | [sparse_rcnn_r50_fpn_300_proposals_crop_mstrain_480-800_3x_coco](https://ml.azure.com/registries/azureml/models/sparse_rcnn_r50_fpn_300_proposals_crop_mstrain_480-800_3x_coco/version/3) | azureml registry |
    | [sparse_rcnn_r101_fpn_300_proposals_crop_mstrain_480-800_3x_coco](https://ml.azure.com/registries/azureml/models/sparse_rcnn_r101_fpn_300_proposals_crop_mstrain_480-800_3x_coco/version/3) | azureml registry |
    | [vfnet_r50_fpn_mdconv_c3-c5_mstrain_2x_coco](https://ml.azure.com/registries/azureml/models/vfnet_r50_fpn_mdconv_c3-c5_mstrain_2x_coco/version/3) | azureml registry |
    | [vfnet_x101_64x4d_fpn_mdconv_c3-c5_mstrain_2x_coco](https://ml.azure.com/registries/azureml/models/vfnet_x101_64x4d_fpn_mdconv_c3-c5_mstrain_2x_coco/version/3) | azureml registry |
    | [yolof_r50_c5_8x8_1x_coco](https://ml.azure.com/registries/azureml/models/yolof_r50_c5_8x8_1x_coco/version/3) | azureml registry |
    | [mask_rcnn_swin-t-p4-w7_fpn_1x_coco](https://ml.azure.com/registries/azureml/models/mask_rcnn_swin-t-p4-w7_fpn_1x_coco/version/3) | azureml registry |

    Below is the folder structure of a registered MLFlow model.

    ![Mlflow Model Tree](../../images/mmd_mlflow_model.png)

    - In the finetuned model, all the augmentations should be specified in artifacts/augmentations.yaml file.
    - All the model files should be stored in artifacts/<model_name>*.pth or artifacts/<model_name>*.py file.
    - **`MLmodel`** is a yaml file and this should contain relavant information. See the sample MLmodel file [here](../../sample_files/MMDMLmodel). Please note that the
    <PYTORCH_MODEL_FOLDER_MOUNTPOINT> is kept as a placeholder only and in the your MLmodel file, you should see an absolute path to the artifact. 

    > Currently _resume_from_checkpoint_ is **NOT** fully enabled with _mlflow_model_. Only the saved model weights can be reloaded but not the optimizer, scheduler and random states

**NOTE** The _pytorch_model_ take priority over _mlflow_model_, in case both inputs are passed


# 2. Outputs
1. _output_dir_ (URI_FOLDER):

    Path to output directory which contains the component metadata and the copied model data.


# 3. Parameters
1. _model_family_ (string, required)

    Which framework the model belongs to.
    It could be one of [`MmDetectionImage`]

2. _model_name_ (string, optional)

    Please select models from AzureML Model Assets for all supported models. For MMDetection models, which are not supported in AzureML model registry, the model's config name is required, same as it's specified in MMDetection Model Zoo. For e.g. fast_rcnn_r101_fpn_1x_coco for [this config file](https://github.com/open-mmlab/mmdetection/blob/master/configs/fast_rcnn/fast_rcnn_r101_fpn_1x_coco.py). You can see the comprehensive list of model configs [here] (https://github.com/open-mmlab/mmdetection/tree/master/configs) and the documentation of model zoo [here] (https://github.com/open-mmlab/mmdetection/blob/main/docs/en/model_zoo.md).
    Please note that it is the user responsibility to comply with the model's license terms.

# 4. Run Settings

This setting helps to choose the compute for running the component code. For the purpose of model selector, cpu compute should work. We recommend using D12 compute.

1. Option1: *Use default compute target*

    If this option is selected, it will identify the compute from setting tab on top right as shown in the below figure
    ![default compute target](../../images/default_compute_from_settings_for_image_components.png)

2. Option2: *Use other compute target*

    - Under this option, you can select either `compute_cluster` or `compute_instance` as the compute type and select any of the already created compute in your workspace.
    - If you have not created the compute, you can create the compute by clicking the `Create Azure ML compute cluster` link that's available while selecting the compute. See the figure below.
    ![other compute target](../../images/other_compute_target_for_image_components.png)
