# description: train Huggingface transformer using DeepSpeed
#
# In this example we train a 1.6B parameter gpt2 model using Deepspeed and
# Huggingface's transformers library.

from dataclasses import dataclass, asdict
from pathlib import Path

from azureml.core import Workspace, ScriptRunConfig, Environment, Experiment
from azureml.core.runconfig import PyTorchConfiguration


TARGET_GPU_COUNT = {
    "gpu-V100-1": 1,
    "gpu-V100-2": 2,
    "gpu-V100-4": 4,
}


@dataclass
class JobArguments:
    """Arguments controlling job submission to Azure ML."""

    target_name: str
    model_checkpoint: str = "distilbert-base-uncased"
    task: str = "cola"
    node_count: int = 1
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 16
    per_device_eval_batch_size: int = 16


def submit_azureml_run(args: JobArguments):
    """Submit GLUE experiment to azureml."""
    ws = Workspace.from_config()

    # get root of git repo
    prefix = Path(__file__).parent
    source_directory = str(prefix.joinpath("src"))

    target = ws.compute_targets[args.target_name]

    env = get_azureml_environment()

    distributed_job_config = get_distributed_job_config(args)

    cmd = f"""ds_report && python finetune_glue.py
    --output_dir outputs
    --model_checkpoint {args.model_checkpoint}
    --task {args.task}
    --num_train_epochs {args.num_train_epochs}
    --per_device_train_batch_size {args.per_device_train_batch_size}
    --per_device_eval_batch_size {args.per_device_eval_batch_size}
    --disable_tqdm 1
    --local_rank $LOCAL_RANK
    --deepspeed ds_config.json
    """.split()

    config = ScriptRunConfig(
        source_directory=source_directory,
        command=cmd,
        environment=env,
        compute_target=target,
        distributed_job_config=distributed_job_config,
    )

    run = Experiment(ws, "deepspeed-transformers-example").submit(config)
    print(run.get_portal_url())  # link to ml.azure.com

    run.set_tags(asdict(args))


def get_azureml_environment():
    env = Environment("deepspeed-transformers")
    env.docker.base_image = None
    env.docker.base_dockerfile = "dockerfile"
    env.python.user_managed_dependencies = True
    env.python.interpreter_path = "/opt/miniconda/bin/python"
    return env


def get_distributed_job_config(args: JobArguments):
    n_proc_per_node = TARGET_GPU_COUNT[args.target_name]
    process_count = n_proc_per_node * args.node_count
    distributed_job_config = PyTorchConfiguration(
        process_count=process_count, node_count=args.node_count
    )
    return distributed_job_config


if __name__ == "__main__":

    target_names = [
        # "gpu-V100-1",  # single GPU
        # "gpu-V100-2",  # two GPUs
        "gpu-V100-4",  # four GPUs
    ]

    # https://huggingface.co/transformers/pretrained_models.html
    model_checkpoints = [
        "distilbert-base-uncased",  # 66M
        # "bert-base-uncased",  # 110M
        # "bert-large-uncased",  # 336M
        # "gpt2",  # 117M
        # "gpt2-medium",  # 345M
        # "gpt2-large",  # 774M
        # "gpt2-xl",  # 1558M
    ]

    # https://openreview.net/pdf?id=rJ4km2R5t7
    tasks = [
        # "wnli",  # 634, inference
        # "rte",  # 2.5k, inference
        # "mrpc",  # 3.7k, paraphrase
        # "stsb",  # 7k, sentence similarity
        "cola",  # 8.5k, single-sentence
        # "sst2",  # 67k, single-sentence
        # "qnli",  # 105k, inference
        # "mnli",  # 393k, inference
        # "qqp",  # 364k, paraphrase
    ]

    for target_name in target_names:
        for model_checkpoint in model_checkpoints:
            for task in tasks:

                args = JobArguments(
                    target_name=target_name,
                    model_checkpoint=model_checkpoint,
                    task=task,
                )

                submit_azureml_run(args)
