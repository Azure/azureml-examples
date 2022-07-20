import os
import onnx
import torch
import mlflow
import logging
import argparse
from azureml.core import Run
from mlflow.tracking.client import MlflowClient
from torchvision.ops._register_onnx_ops import _onnx_opset_version
from azureml.automl.dnn.vision.common.logging_utils import get_logger
from azureml.automl.dnn.vision.common.model_export_utils import load_model
import azureml.automl.core.shared.constants as shared_constants
from azureml.automl.dnn.vision.object_detection_yolo.models.common import (
    Conv,
    Hardswish,
)

logger = get_logger(__name__)
logging.basicConfig(level=logging.INFO)


def export_onnx_model(
    model, dummy_input, input_names, output_names, dynamic_axes, file_path, device
):
    """
    Export the pytorch model to onnx model file.

    :param model: Model object for faster_rcnn or yolo or mask_rcnn
    :type model: torchvision.models.detection
    :param dummy_input: dummy input to be used for ONNX model
    :type dummy_input: torch tensor
    :param input_names: Input names to be used for ONNX model
    :type input_names: dict
    :param output_names: Output names to be used for ONNX model
    :type output_names: dict
    :param dynamic_axes: Dynamic axes to be used for ONNX model
    :type dynamic_axes: dict
    :param file_path: File path to save the exported onnx model
    :type file_path: str
    :param device: Device where model should be run
    :type device: str
    """

    model.eval()
    model.to(device)
    torch.onnx.export(
        model,
        dummy_input,
        file_path,
        opset_version=_onnx_opset_version,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
    )


def generate_onnx_batch_model(
    batch_size,
    height_onnx,
    width_onnx,
    task_type,
    best_child_run,
    mlflow_client,
    model_settings,
    model_name,
    device,
):
    """
    Get the onnx batch model and verify the schema.

    :param batch_size: Batch size needed for batch inference
    :type batch_size: int
    :param height_onnx: Height of the image to be used for ONNX model
    :type height_onnx: int
    :param width_onnx: Width of the image to be used for ONNX model
    :type width_onnx: int
    :param task_type: Type of vision task from image classification,
     object detection, segmentation
    :type task_type: str
    :param best_child_run: Best child run object of an experiment
    :type best_child_run: azureml.core.run.Run
    :param mlflow_client: mlflow client of an experiment
    :type mlflow_client: mlflow.tracking.client.MlflowClient
    :param model_settings: Model parameters with key, values
    :type model_settings: dict
    :param model_type: Model name
    :type model_type: str
    :param device: Device on which the model has to be generated
    :type device: str
    """

    # download the pytorch model weights
    remote_path = os.path.join(
        shared_constants.OUTPUT_PATH, shared_constants.PT_MODEL_FILENAME
    )
    local_path = mlflow_client.download_artifacts(
        best_child_run.info.run_id, remote_path, "./"
    )
    logger.info("local path of downloaded model: {}".format(local_path))

    # load the model wrapper
    model_wrapper = load_model(task_type, local_path, **model_settings)
    onnx_model_file_path = os.path.join(
        shared_constants.OUTPUT_PATH, "model_" + str(batch_size) + ".onnx"
    )

    if batch_size <= 1:
        msg = "Please use the auto-generated ONNX model for the best child run.\
               No need to run this script"
        logger.warning(msg)
        return

    input_names = ["input"]
    dummy_input = torch.randn(batch_size, 3, height_onnx, width_onnx).to(device)

    if model_name.startswith("faster") or model_name.startswith("retina"):

        if model_name.startswith("retina"):
            od_output_names = ["boxes", "scores", "labels"]
        else:
            od_output_names = ["boxes", "labels", "scores"]
        output_names = [
            name + "_" + str(sample_id)
            for sample_id in range(batch_size)
            for name in od_output_names
        ]
        dynamic_axes = dict()
        dynamic_axes["input"] = {0: "batch", 1: "channel", 2: "height", 3: "width"}
        for output in output_names:
            dynamic_axes[output] = {0: "prediction"}

        model_wrapper.disable_model_transform()
        model = model_wrapper.model

    elif model_name.startswith("yolo"):

        output_names = ["output"]
        dynamic_axes = dict()
        dynamic_axes = {"input": {0: "batch"}, "output": {0: "batch", 1: "boxes"}}

        model = model_wrapper.model
        for k, m in model.named_modules():
            if isinstance(m, Conv) and isinstance(m.act, torch.nn.Hardswish):
                m.act = Hardswish()

    elif model_name.startswith("mask"):

        od_output_names = ["boxes", "labels", "scores", "masks"]
        output_names = [
            name + "_" + str(sample_id)
            for sample_id in range(batch_size)
            for name in od_output_names
        ]
        dynamic_axes = dict()
        dynamic_axes["input"] = {0: "batch", 1: "channel", 2: "height", 3: "width"}
        for output in output_names:
            dynamic_axes[output] = {0: "prediction"}
        masks_output_names = [name for name in output_names if "masks" in name]
        for mask_name in masks_output_names:
            dynamic_axes[mask_name][2] = "height"
            dynamic_axes[mask_name][3] = "width"
        model_wrapper.disable_model_transform()
        model = model_wrapper.model

    try:
        # export the model
        export_onnx_model(
            model,
            dummy_input,
            input_names,
            output_names,
            dynamic_axes,
            onnx_model_file_path,
            device=device,
        )
        # check/verify schema of generated onnx model
        onnx_model = onnx.load(onnx_model_file_path)
        onnx.checker.check_model(onnx_model)
        logger.info(
            "ONNX model generation for the batch size {} is successful".format(
                batch_size
            )
        )
    except Exception as e:
        logger.error("ONNX model generation or Schema validation error: " + str(e))
        raise


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    # add arguments
    parser.add_argument(
        "--model_name",
        type=str,
        required=True,
        help="Model name supported from yolov5, fasterrcnn_resnet152_fpn,\
                        maskrcnn_resnet101_fpn etc.",
    )
    parser.add_argument(
        "--batch_size", type=int, required=True, help="No. of samples per batch"
    )
    parser.add_argument(
        "--height_onnx",
        type=int,
        required=True,
        help="Height of the image for ONNX model",
    )
    parser.add_argument(
        "--width_onnx",
        type=int,
        required=True,
        help="Width of the image for ONNX model",
    )
    parser.add_argument(
        "--job_name", type=str, required=True, help="job name of the run"
    )
    parser.add_argument(
        "--task_type", type=str, required=True, help="Task type in automl for images"
    )
    parser.add_argument(
        "--box_score_thresh",
        type=float,
        required=True,
        help="During inference, only return proposals with a \
              classification score greater than box_score_thresh",
    )

    parser.add_argument(
        "--min_size",
        type=int,
        required=False,
        help="Minimum size of the image to be rescaled before feeding it to \
        the backbone",
    )
    parser.add_argument(
        "--max_size",
        type=int,
        required=False,
        help="Maximum size of the image to be rescaled before feeding it to \
        the backbone",
    )
    parser.add_argument(
        "--box_nms_thresh",
        type=float,
        required=False,
        help="Non-maximum suppression (NMS) threshold for the prediction head",
    )
    parser.add_argument(
        "--box_detections_per_img",
        type=int,
        required=False,
        help="Maximum number of detections per image, for all classes",
    )
    parser.add_argument(
        "--img_size",
        type=int,
        required=False,
        help="Image size for train and validation",
    )
    parser.add_argument(
        "--model_size",
        type=str,
        required=False,
        help="Size of the model from small, medium, large, xlarge",
    )
    parser.add_argument(
        "--box_iou_thresh", type=float, required=False, help="IoU threshold"
    )

    # parse arguments
    args = parser.parse_args()
    experiment = Run.get_context().experiment
    mlflow_tracking_uri = experiment.workspace.get_mlflow_tracking_uri()
    logger.info("mlflow_tracking_uri: {0}".format(mlflow_tracking_uri))
    mlflow.set_tracking_uri(mlflow_tracking_uri)
    mlflow.set_experiment(experiment.name)

    # Initialize MLFlow client
    mlflow_client = MlflowClient()

    # Get the parent run
    mlflow_parent_run = mlflow_client.get_run(args.job_name)
    logger.info("Parent Run: {} ".format(args.job_name))

    # Get the best model's child run
    best_child_run_id = mlflow_parent_run.data.tags["automl_best_child_run_id"]
    logger.info("Found best child run id: {}".format(best_child_run_id))

    best_child_run = mlflow_client.get_run(best_child_run_id)

    model_type = None
    if args.task_type == "image-object-detection":
        if args.model_name.startswith("yolo"):
            # yolo settings
            model_settings = {
                "img_size": args.img_size,
                "model_size": args.model_size,
                "box_score_thresh": args.box_score_thresh,
                "box_iou_thresh": args.box_iou_thresh,
            }

        elif args.model_name.startswith("faster") or args.model_name.startswith(
            "retina"
        ):
            # faster rcnn settings
            model_settings = {
                "min_size": args.min_size,
                "max_size": args.max_size,
                "box_score_thresh": args.box_score_thresh,
                "box_nms_thresh": args.box_nms_thresh,
                "box_detections_per_img": args.box_detections_per_img,
            }
        else:
            logger.info(
                "Given model name {} for the task {} is not expected".format(
                    args.model_name, args.task_type
                )
            )
    elif args.task_type == "image-instance-segmentation":
        # mask rcnn settings
        if args.model_name.startswith("mask"):
            model_settings = {
                "min_size": args.min_size,
                "max_size": args.max_size,
                "box_score_thresh": args.box_score_thresh,
                "box_nms_thresh": args.box_nms_thresh,
                "box_detections_per_img": args.box_detections_per_img,
            }
        else:
            logger.info(
                "Given model name {} for the task {} is not expected".format(
                    args.model_name, args.task_type
                )
            )

    #     if model_type is not None:
    generate_onnx_batch_model(
        args.batch_size,
        args.height_onnx,
        args.width_onnx,
        args.task_type,
        best_child_run,
        mlflow_client,
        model_settings,
        args.model_name,
        device="cpu",
    )
