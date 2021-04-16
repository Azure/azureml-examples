# description: Automatic hyperparameter optimization with Azure ML HyperDrive library.

"""Automatic hyperparameter optimization with Azure ML HyperDrive library.

This submits a HyperDrive experiment to optimize for a set of hyperparameters.
We use:

- Early termination policy to halt "poorly performing" runs
- Concurrency, that allows us to parellelize individual finetuning runs
"""
import argparse
import numpy as np
from pathlib import Path
from azureml.core import Workspace  # connect to workspace
from azureml.core import ComputeTarget  # specify AzureML compute resources
from azureml.core import Experiment  # connect/create experiments
from azureml.core import Environment  # manage e.g. Python environments
from azureml.core import ScriptRunConfig  # prepare code, an run configuration

# hyperdrive imports
from azureml.train.hyperdrive import (
    RandomParameterSampling,
    BayesianParameterSampling,
    TruncationSelectionPolicy,
    MedianStoppingPolicy,
    HyperDriveConfig,
)
from azureml.train import hyperdrive


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
    target: ComputeTarget = ws.compute_targets["gpu-K80-2"]
    env: Environment = transformers_environment(use_gpu=True)

    # set up script run configuration
    config = ScriptRunConfig(
        source_directory=str(Path(__file__).parent.joinpath("src")),
        script="finetune_glue.py",
        arguments=[
            "--output_dir",
            "outputs",
            "--task",
            args.glue_task,
            "--model_checkpoint",
            args.model_checkpoint,
            # training args
            "--evaluation_strategy",
            "steps",  # more frequent evaluation helps HyperDrive
            "--eval_steps",
            200,
            "--learning_rate",
            2e-5,  # will be overridden by HyperDrive
            "--per_device_train_batch_size",
            16,  # will be overridden by HyperDrive
            "--per_device_eval_batch_size",
            16,
            "--num_train_epochs",
            5,
            "--weight_decay",
            0.01,  # will be overridden by HyperDrive
            "--disable_tqdm",
            True,
        ],
        compute_target=target,
        environment=env,
    )

    # set up hyperdrive search space
    convert_base = lambda x: float(np.log(x))
    search_space = {
        "--learning_rate": hyperdrive.loguniform(
            convert_base(1e-6), convert_base(5e-2)
        ),  # NB. loguniform on [exp(min), exp(max)]
        "--weight_decay": hyperdrive.uniform(5e-3, 15e-2),
        "--per_device_train_batch_size": hyperdrive.choice([16, 32]),
    }

    hyperparameter_sampling = RandomParameterSampling(search_space)

    policy = TruncationSelectionPolicy(
        truncation_percentage=50, evaluation_interval=2, delay_evaluation=0
    )

    hyperdrive_config = HyperDriveConfig(
        run_config=config,
        hyperparameter_sampling=hyperparameter_sampling,
        policy=policy,
        primary_metric_name="eval_matthews_correlation",
        primary_metric_goal=hyperdrive.PrimaryMetricGoal.MAXIMIZE,
        max_total_runs=20,
        max_concurrent_runs=8,
    )

    run = Experiment(ws, "transformers-glue-finetuning-hyperdrive").submit(
        hyperdrive_config
    )
    print(run.get_portal_url())
    run.wait_for_completion(show_output=True)
