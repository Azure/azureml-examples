# description: Experiment comparing training performance of GLUE finetuning task with differing hardware.

"""Experiment comparing training performance of GLUE finetuning task with differing hardware.

This script prepares the `src/finetune_glue.py` script to run in Azure ML using
different compute clusters. The idea of this experiment is to compare training
times between different VM SKUs.

To run this script you need:

    - An Azure ML Workspace
    - A ComputeTarget to train on (we recommend a GPU-based compute cluster)
    - Azure ML Environment:
        - create the required python environment by running the `aml_utils.py` script
        - This registers two environments "transformers-datasets-cpu" and "transformers-datasets-gpu"

Note:
    
    Arguments passed to `src/finetune_glue.py` will override TrainingArguments:
    
    https://huggingface.co/transformers/main_classes/trainer.html#trainingarguments

"""
import argparse
from pathlib import Path
from azureml.core import Workspace  # connect to workspace
from azureml.core import ComputeTarget  # specify AzureML compute resources
from azureml.core import Experiment  # connect/create experiments
from azureml.core import Environment  # manage e.g. Python environments
from azureml.core import ScriptRunConfig  # prepare code, an run configuration
from azureml.core import Run  # used for type hints


def transformers_environment(use_gpu=True):
    """Prepares Azure ML Environment with transformers library.

    Note: We install transformers library from source. See requirements file for
    full list of dependencies.

    Args:
        use_gpu (bool): If true, Azure ML will use gpu-enabled docker image
            as base.

    Return:
        Azure ML Environment with huggingface libraries needed to perform GLUE
        finetuning task.
    """

    pip_requirements_path = str(Path(__file__).parent.joinpath("requirements.txt"))
    print(f"Create Azure ML Environment from {pip_requirements_path}")

    if use_gpu:

        env_name = "transformers-gpu"
        env = Environment.from_pip_requirements(
            name=env_name,
            file_path=pip_requirements_path,
        )
        env.docker.base_image = (
            "mcr.microsoft.com/azureml/intelmpi2018.3-cuda10.0-cudnn7-ubuntu16.04"
        )

    else:

        env_name = "transformers-cpu"
        env = Environment.from_pip_requirements(
            name=env_name,
            file_path=pip_requirements_path,
        )

    return env


def submit_glue_finetuning_to_aml(
    glue_task: str,
    model_checkpoint: str,
    environment: Environment,
    target: ComputeTarget,
    experiment: Experiment,
) -> Run:
    """Submit GLUE finetuning task to Azure ML.

    This method prepares the configuration (compute target and environment) together
    with the training code (see src) into a ScriptRunConfig, and submits it to Azure
    ML.

    Args:
        glue_task (str): Name of the GLUE finetuning task. One of: "cola", "mnli",
            "mnli-mm", "mrpc", "qnli", "qqp", "rte", "sst2", "stsb", "wnli".
        model_checkpoint (str): Name of the transformers pretrained model to use
            for finetuning. See https://huggingface.co/transformers/pretrained_models.html
        environment (Environment): The Azure ML environment to use.
        target (ComputeTarget):  The Azure ML compute target to train on.
        experiment (Experiment):  The Azure ML experiment used to submit the run.

    Return:
        The Azure ML Run instance associated to this finetuning submission.
    """
    # set up script run configuration
    config = ScriptRunConfig(
        source_directory=str(Path(__file__).parent.joinpath("src")),
        script="finetune_glue.py",
        arguments=[
            "--output_dir",
            "outputs",
            "--task",
            glue_task,
            "--model_checkpoint",
            model_checkpoint,
            # training args
            "--num_train_epochs",
            5,
            "--learning_rate",
            2e-5,
            "--per_device_train_batch_size",
            16,
            "--per_device_eval_batch_size",
            16,
            "--disable_tqdm",
            True,
        ],
        compute_target=target,
        environment=environment,
    )

    # submit script to AML
    run = experiment.submit(config)
    run.set_tags(
        {
            "task": glue_task,
            "target": target.name,
            "environment": environment.name,
            "model": model_checkpoint,
        }
    )

    return run


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--glue_task", default="cola", help="Name of GLUE task used for finetuning."
    )
    parser.add_argument(
        "--model_checkpoint",
        default="distilbert-base-uncased",
        help="Pretrained transformers model name.",
    )
    args = parser.parse_args()

    print(
        f"Finetuning {args.glue_task} with model {args.model_checkpoint} on Azure ML..."
    )

    # get Azure ML resources
    ws: Workspace = Workspace.from_config()
    env: Environment = transformers_environment(use_gpu=True)
    exp: Experiment = Experiment(ws, "transformers-glue-finetuning-sku-comparison")

    runs = []

    target_names = ["gpu-cluster", "gpu-K80-2"]
    for target_name in target_names:

        target: ComputeTarget = ws.compute_targets[target_name]

        run: Run = submit_glue_finetuning_to_aml(
            glue_task=args.glue_task,  # one of: "cola", "mnli", "mnli-mm", "mrpc", "qnli", "qqp", "rte", "sst2", "stsb", "wnli"
            model_checkpoint=args.model_checkpoint,  # try: "bert-base-uncased"
            environment=env,
            target=target,
            experiment=exp,
        )
        runs.append(run)

        print(f"Submitted to {target.name}: {run.get_portal_url()}\n")

    for run in runs:
        run.wait_for_completion(show_output=True)
