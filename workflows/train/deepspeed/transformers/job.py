from dataclasses import dataclass, asdict

from azureml.core import Workspace, ScriptRunConfig, Environment, Experiment
from azureml.core.runconfig import MpiConfiguration


TARGET_GPU_COUNT = {
        "gpu-V100-1": 1,
        "gpu-V100-2": 2,
        "gpu-V100-4": 4,
    }

@dataclass
class DeepspeedExperimentArguments:
    target_name: str
    use_deepspeed: bool = True
    model_checkpoint: str = "distilbert-base-uncased"
    task: str = "cola"
    node_count: int = 1

def submit_azureml_run(args: DeepspeedExperimentArguments):
    """Submit GLUE experiment to azureml."""
    ws = Workspace.from_config()

    target = ws.compute_targets[args.target_name]

    env = get_azureml_environment()

    distributed_job_config = get_distributed_job_config(args)

    cmd = build_command(args)
    
    config = ScriptRunConfig(
        source_directory="t5",
        command=cmd,
        environment=env,
        compute_target=target,
        distributed_job_config=distributed_job_config,
    )

    run = Experiment(ws, 'deepspeed-t5').submit(config)
    print(run.get_portal_url()) # link to ml.azure.com

    run.set_tags(asdict(args))

def get_azureml_environment():
    env = Environment("deepspeed-transformers")
    env.docker.base_image = None
    env.docker.base_dockerfile = "dockerfile"
    env.python.user_managed_dependencies=True
    env.python.interpreter_path = "/opt/miniconda/bin/python"
    return env

def get_distributed_job_config(args: DeepspeedExperimentArguments):
    n_proc = TARGET_GPU_COUNT[args.target_name]
    distributed_job_config = MpiConfiguration(process_count_per_node=n_proc, node_count=args.node_count)
    return distributed_job_config

def build_command(args: DeepspeedExperimentArguments):
    cmd = f"""nvidia-smi && python finetune_glue.py
    --output_dir outputs
    --model_checkpoint {args.model_checkpoint}
    --task {args.task}
    --num_train_epochs 5
    --learning_rate 2e-5
    --per_device_train_batch_size 16
    --per_device_eval_batch_size 16
    --disable_tqdm 1
    """.split()

    if args.use_deepspeed:

        cmd += """
        --deepspeed ds_config.json
        --local_rank $AZ_BATCHAI_TASK_INDEX
        """.split()
    
    return cmd

if __name__ == "__main__":

    target_names = [
        # "gpu-V100-1",  # single GPU
        # "gpu-V100-2",  # two GPUs
        "gpu-V100-4",  # 4 GPUs
    ]
    
    # https://huggingface.co/transformers/pretrained_models.html
    model_checkpoints = [
        # "distilbert-base-uncased",  # 66M
        # "bert-base-uncased",  # 110M
        # "bert-large-uncased",  # 336M
        # "gpt2",  # 117M
        # "gpt2-medium",  # 345M
        # "gpt2-large",  # 774M
        "gpt2-xl",  # 1558M
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
                for use_deepspeed in [True, False]:
                    
                    args = DeepspeedExperimentArguments(
                        target_name=target_name,
                        use_deepspeed=use_deepspeed,
                        model_checkpoint=model_checkpoint,
                        task=task,
                    )

                    submit_azureml_run(args)
