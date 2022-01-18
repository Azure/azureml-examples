import torch
import onnx
import argparse
from azureml.train.automl.run import AutoMLRun
from azureml.core import Experiment
from azureml.core.workspace import Workspace
from azureml.automl.dnn.vision.common.model_export_utils import load_model
from torchvision.ops._register_onnx_ops import _onnx_opset_version
from azureml.automl.dnn.vision.common.logging_utils import get_logger

logger = get_logger(__name__)

def export_onnx_model(model_wrapper, batch_size, height_onnx, width_onnx, file_path, device):
    """
    Export the pytorch model to onnx model file.

    :param model_wrapper: Model wrapper containing model for MaskRCNN
    :type model_wrapper: object_detection.models.object_detection_model_wrappers.ModelWrapper
    :param batch_size: Batch size needed for batch inference
    :type batch_size: int
    :param height_onnx: Height of the image to be used for ONNX model
    :type height_onnx: int
    :param width_onnx: Width of the image to be used for ONNX model
    :type width_onnx: int
    :param file_path: file path to save the exported onnx model
    :type file_path: str
    :param device: device where model should be run (usually 'cpu' or 'cuda:0' if it is the first gpu)
    :type device: str
    """

    input_names = ['input']
    od_output_names = ['boxes', 'labels', 'scores', 'masks']
    output_names = [name + "_" + str(sample_id) for sample_id in range(batch_size) for name in od_output_names]
    dynamic_axes = dict()
    dynamic_axes['input'] = {0: 'batch'}
    for output in output_names:
        dynamic_axes[output] = {0: 'prediction'}

    dummy_input = torch.randn(batch_size, 3, height_onnx, width_onnx).to(device)
    model_wrapper.disable_model_transform()
    model = model_wrapper.model
    model.eval()
    model.to(device)
    torch.onnx.export(model,
                      dummy_input,
                      file_path,
                      opset_version=_onnx_opset_version,
                      input_names=input_names,
                      output_names=output_names,
                      dynamic_axes=dynamic_axes)

def generate_onnx_batch_model(batch_size, height_onnx, width_onnx, task_type, best_child_run, model_settings):
    """
    Get the onnx batch model and verify the schema.

    :param batch_size: Batch size needed for batch inference
    :type batch_size: int
    :param height_onnx: Height of the image to be used for ONNX model
    :type height_onnx: int
    :param width_onnx: Width of the image to be used for ONNX model
    :type width_onnx: int
    :param task_type: Type of vision task from image classificatio, object detection, segmentation
    :type task_type: str
    :param best_child_run: best child run object of an experiment
    :type best_child_run: azureml.core.run.Run
    :param model_settings: model parameters with key, values
    :type model_settings: dict
    """

    # download the pytorch model weights
    torch_model_path = "model.pt"
    best_child_run.download_file(name="train_artifacts/model.pt", output_file_path=torch_model_path)
    # load the model wrapper
    model_wrapper = load_model(task_type, torch_model_path, **model_settings)

    if (batch_size > 1) or (height_onnx != 600) or (width_onnx != 800):
        onnx_model_file_path = "./outputs/model_" + str(batch_size) + ".onnx"
        export_onnx_model(model_wrapper, batch_size, height_onnx, width_onnx, onnx_model_file_path, device='cpu')
    else:
        msg = "Please use the auto-generated ONNX model for the best child run. No need to run this script"
        logger.warning(msg)
    try:
        # check/verify schema of generated onnx model
        onnx_model = onnx.load(onnx_model_file_path)
        onnx.checker.check_model(onnx_model)
        logger.info("ONNX model generation for the batch size {} is successful".format(batch_size))
    except Exception as e:
        logger.error("ONNX model Schema validation error: " + str(e))
        raise

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    # add arguments
    parser.add_argument('--batch_size', type=int, required=True)
    parser.add_argument('--height_onnx', type=int, required=True)
    parser.add_argument('--width_onnx', type=int, required=True)
    parser.add_argument('--experiment_name', type=str, required=True)
    parser.add_argument('--subscription_id', type=str, required=True)
    parser.add_argument('--resource_group', type=str, required=True)
    parser.add_argument('--workspace_name', type=str, required=True)
    parser.add_argument('--run_id', type=str, required=True)
    parser.add_argument('--task_type', type=str, required=True)
    parser.add_argument('--min_size', type=int, required=True)
    parser.add_argument('--max_size', type=int, required=True)
    parser.add_argument('--box_score_thresh', type=float, required=True)
    parser.add_argument('--box_nms_thresh', type=float, required=True)
    parser.add_argument('--box_detections_per_img', type=int, required=True)
    # parse arguments
    args = parser.parse_args()

    ws = Workspace.create(name=args.workspace_name,
                          subscription_id=args.subscription_id,
                          resource_group=args.resource_group,
                          exist_ok=True)
    experiment = Experiment(ws, name=args.experiment_name)

    # load the best child
    automl_image_run = AutoMLRun(experiment=experiment, run_id=args.run_id)
    best_child_run = automl_image_run.get_best_child()

    model_settings = {"min_size": args.min_size, "max_size": args.max_size, "box_score_thresh": args.box_score_thresh,
                      "box_nms_thresh": args.box_nms_thresh, "box_detections_per_img": args.box_detections_per_img}

    generate_onnx_batch_model(args.batch_size, args.height_onnx, args.width_onnx, args.task_type, best_child_run,
                              model_settings)
