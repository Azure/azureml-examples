import onnx
import torch
import argparse
from azureml.core import Experiment
from azureml.core.workspace import Workspace
from azureml.train.automl.run import AutoMLRun
from torchvision.ops._register_onnx_ops import _onnx_opset_version
from azureml.automl.dnn.vision.common.logging_utils import get_logger
from azureml.automl.dnn.vision.common.model_export_utils import load_model
from azureml.automl.dnn.vision.object_detection_yolo.models.common import Conv, Hardswish

logger = get_logger(__name__)

def export_onnx_model(model, dummy_input, input_names, output_names, dynamic_axes, file_path, device):
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
    torch.onnx.export(model,
                      dummy_input,
                      file_path,
                      opset_version=_onnx_opset_version,
                      input_names=input_names,
                      output_names=output_names,
                      dynamic_axes=dynamic_axes)

def generate_onnx_batch_model(batch_size, height_onnx, width_onnx, task_type, best_child_run,
                              model_settings, model_type, device):
    """
    Get the onnx batch model and verify the schema.

    :param batch_size: Batch size needed for batch inference
    :type batch_size: int
    :param height_onnx: Height of the image to be used for ONNX model
    :type height_onnx: int
    :param width_onnx: Width of the image to be used for ONNX model
    :type width_onnx: int
    :param task_type: Type of vision task from image classification, object detection, segmentation
    :type task_type: str
    :param best_child_run: Best child run object of an experiment
    :type best_child_run: azureml.core.run.Run
    :param model_settings: Model parameters with key, values
    :type model_settings: dict
    :param model_type: Model name
    :type model_type: str
    :param device: Device on which the model has to be generated
    :type device: str
    """

    # download the pytorch model weights
    torch_model_path = 'model.pt'
    best_child_run.download_file(name='train_artifacts/model.pt', output_file_path=torch_model_path)
    # load the model wrapper
    model_wrapper = load_model(task_type, torch_model_path, **model_settings)
    onnx_model_file_path = './outputs/model_' + str(batch_size) + '.onnx'
    batch_model = False  # to check whether the user is not generating the default onnx model

    if model_type == 'faster_rcnn':

        if (batch_size > 1) or (height_onnx != 600) or (width_onnx != 800):
            batch_model = True
            input_names = ['input']
            od_output_names = ['boxes', 'labels', 'scores']
            output_names = [name + "_" + str(sample_id) for sample_id in range(batch_size) for name in od_output_names]
            dynamic_axes = dict()
            dynamic_axes['input'] = {0: 'batch'}
            for output in output_names:
                dynamic_axes[output] = {0: 'prediction'}

            dummy_input = torch.randn(batch_size, 3, height_onnx, width_onnx).to(device)
            model_wrapper.disable_model_transform()
            model = model_wrapper.model

    elif model_type == 'yolo':

        if (batch_size > 1) or (height_onnx != 640) or (width_onnx != 640):
            batch_model = True
            input_names = ['input']
            output_names = ['output']
            dynamic_axes = dict()
            dynamic_axes = {'input': {0: 'batch'}, 'output': {0: 'batch', 1: 'boxes'}}

            dummy_input = torch.randn(batch_size, 3, height_onnx, width_onnx).to(device)
            model = model_wrapper.model
            for k, m in model.named_modules():
                if isinstance(m, Conv) and isinstance(m.act, torch.nn.Hardswish):
                    m.act = Hardswish()

    elif model_type == 'mask_rcnn':

        if (batch_size > 1) or (height_onnx != 600) or (width_onnx != 800):
            batch_model = True
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

    if batch_model:
        try:
            # export the model
            export_onnx_model(model, dummy_input, input_names, output_names, dynamic_axes,
                              onnx_model_file_path, device=device)
            # check/verify schema of generated onnx model
            onnx_model = onnx.load(onnx_model_file_path)
            onnx.checker.check_model(onnx_model)
            logger.info('ONNX model generation for the batch size {} is successful'.format(batch_size))
        except Exception as e:
            logger.error('ONNX model generation or Schema validation error: ' + str(e))
            raise
    else:
        msg = 'Please use the auto-generated ONNX model for the best child run. No need to run this script'
        logger.warning(msg)

if __name__ == '__main__':

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
    parser.add_argument('--box_score_thresh', type=float, required=True)

    parser.add_argument('--min_size', type=int, required=False)
    parser.add_argument('--max_size', type=int, required=False)
    parser.add_argument('--box_nms_thresh', type=float, required=False)
    parser.add_argument('--box_detections_per_img', type=int, required=False)
    parser.add_argument('--img_size', type=int, required=False)
    parser.add_argument('--model_size', type=str, required=False)
    parser.add_argument('--box_iou_thresh', type=float, required=False)

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

    model_type = None
    if args.task_type == 'image-object-detection':
        if args.img_size is not None:
            # yolo settings
            model_type = 'yolo'
            model_settings = {"img_size": args.img_size, "model_size": args.model_size,
                              "box_score_thresh": args.box_score_thresh, "box_iou_thresh": args.box_iou_thresh}
        else:
            # faster rcnn settings
            model_type = 'faster_rcnn'
            model_settings = {"min_size": args.min_size, "max_size": args.max_size,
                              "box_score_thresh": args.box_score_thresh,
                              "box_nms_thresh": args.box_nms_thresh,
                              "box_detections_per_img": args.box_detections_per_img}

    elif args.task_type == 'image-instance-segmentation':
        # mask rcnn settings
        model_type = 'mask_rcnn'
        model_settings = {"min_size": args.min_size, "max_size": args.max_size,
                          "box_score_thresh": args.box_score_thresh,
                          "box_nms_thresh": args.box_nms_thresh, "box_detections_per_img": args.box_detections_per_img}

    if model_type is not None:
        generate_onnx_batch_model(args.batch_size, args.height_onnx, args.width_onnx, args.task_type, best_child_run,
                                  model_settings, model_type, device='cpu')
    else:
        logger.info('Given task type {} is not expected'.format(args.task_type))
