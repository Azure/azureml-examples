# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""Direct Reasoning Optimization trainer using verl.trainer.main_ppo."""
import argparse
import json
import socket
import subprocess
import sys
import time
import os

from azureml.acft.common_components import get_logger_app, set_logging_parameters, LoggingLiterals
from azureml.acft.common_components.utils.error_handling.error_definitions import ACFTUserError
from azureml._common._error_definition.azureml_error import AzureMLError
from azureml.acft.common_components.utils.error_handling.exceptions import ACFTValidationException

TASK_TYPE = 'chat-completion'
COMPONENT_NAME = "ACFT-VERL_GRPO"
logger = get_logger_app("azureml.acft.contrib.hf.scripts.src.grpo.reasoning_verl_grpo")


# ============================================================================
# JSONL Support via Custom Dataset Class
# ============================================================================
# This script uses VERL's built-in custom dataset class feature to support
# JSONL format files without modifying the verl codebase.
# ============================================================================


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Direct Reasoning Optimization using verl.trainer.main_ppo")

    # PyPI packages override
    parser.add_argument("--pypi_packages_override", type=str, default=None,
                        help="Comma-separated list of PyPI packages\
                        to override (e.g., transformers==4.30.0,torch==2.3.1)")

    # Engine argument
    parser.add_argument("--ENGINE", type=str, default="vllm", help="Engine type (default: vllm)")

    # Data arguments
    parser.add_argument("--data_train_files", type=str, required=True, help="Path to the training parquet file")
    parser.add_argument("--data_val_files", type=str, required=True, help="Path to the validation parquet file")
    parser.add_argument("--data_train_batch_size", type=int, default=512, help="Training batch size")
    parser.add_argument("--data_max_prompt_length", type=int, default=1024, help="Maximum prompt length")
    parser.add_argument("--data_max_response_length", type=int, default=2048, help="Maximum response length")
    parser.add_argument("--data_filter_overlong_prompts", type=bool, default=True, help="Filter overlong prompts")
    parser.add_argument("--data_truncation", type=str, default="error", help="Truncation strategy")
    parser.add_argument("--data_image_key", type=str, default="images", help="Image key column")

    # Actor model arguments
    parser.add_argument("--actor_model_path", type=str, default="Qwen/Qwen2.5-VL-7B-Instruct", help="Model path")
    parser.add_argument("--actor_optim_lr", type=float, default=3e-6, help="Actor optimizer learning rate")
    parser.add_argument("--actor_model_use_remove_padding", type=bool, default=True, help="Use remove padding in model")
    parser.add_argument("--actor_strategy", type=str, default="fsdp2", help="Actor training strategy (e.g., fsdp, fsdp2)")
    parser.add_argument("--actor_fsdp_config_offload_policy", type=bool, default=True, help="FSDP config offload policy")
    parser.add_argument("--actor_fsdp_config_model_dtype", type=str, default="bf16", help="FSDP config model dtype (currently not used by VERL)")
    parser.add_argument("--actor_fsdp_config_mixed_precision_param_dtype", type=str, default="bf16", help="FSDP config mixed precision param dtype")
    parser.add_argument("--actor_fsdp_config_mixed_precision_reduce_dtype", type=str, default="fp32", help="FSDP config mixed precision reduce dtype")
    parser.add_argument("--actor_fsdp_config_mixed_precision_buffer_dtype", type=str, default="fp32", help="FSDP config mixed precision buffer dtype")
    parser.add_argument("--actor_ppo_mini_batch_size", type=int, default=128, help="PPO mini batch size")
    parser.add_argument("--actor_ppo_micro_batch_size_per_gpu", type=int, default=10, help="PPO micro batch size per GPU")
    parser.add_argument("--actor_model_lora_rank", type=int, default=64, help="LoRA rank")
    parser.add_argument("--actor_model_lora_alpha", type=int, default=32, help="LoRA alpha")
    parser.add_argument("--actor_model_target_modules", type=str, default="all-linear", help="Target modules for LoRA")
    parser.add_argument("--actor_model_exclude_modules", type=str, default=".*visual.*", help="Exclude modules regex")
    parser.add_argument("--actor_use_kl_loss", type=bool, default=True, help="Use KL loss")
    parser.add_argument("--actor_kl_loss_coef", type=float, default=0.01, help="KL loss coefficient")
    parser.add_argument("--actor_kl_loss_type", type=str, default="low_var_kl", help="KL loss type")
    parser.add_argument("--actor_entropy_coeff", type=int, default=0, help="Entropy coefficient")
    parser.add_argument("--actor_model_enable_gradient_checkpointing", type=bool, default=True, help="Enable gradient checkpointing")
    parser.add_argument("--actor_fsdp_param_offload", type=bool, default=False, help="FSDP param offload")
    parser.add_argument("--actor_fsdp_optimizer_offload", type=bool, default=False, help="FSDP optimizer offload")

    # Rollout arguments
    parser.add_argument("--rollout_log_prob_micro_batch_size_per_gpu", type=int, default=20, help="Rollout log prob micro batch size per GPU")
    parser.add_argument("--rollout_tensor_model_parallel_size", type=int, default=2, help="Rollout tensor model parallel size")
    parser.add_argument("--rollout_name", type=str, default="vllm", help="Rollout name (engine)")
    parser.add_argument("--rollout_dtype", type=str, default="float16", help="Rollout data type (e.g., float16, bfloat16, float32)")
    parser.add_argument("--rollout_disable_mm_preprocessor_cache", type=bool, default=True, help="Disable MM preprocessor cache")
    parser.add_argument("--rollout_gpu_memory_utilization", type=float, default=0.6, help="Rollout GPU memory utilization")
    parser.add_argument("--rollout_enable_chunked_prefill", type=bool, default=False, help="Enable chunked prefill")
    parser.add_argument("--rollout_enforce_eager", type=bool, default=False, help="Enforce eager execution")
    parser.add_argument("--rollout_free_cache_engine", type=bool, default=False, help="Free cache engine")
    parser.add_argument("--rollout_n", type=int, default=5, help="Rollout n")

    # Reference arguments
    parser.add_argument("--ref_log_prob_micro_batch_size_per_gpu", type=int, default=20, help="Ref log prob micro batch size per GPU")
    parser.add_argument("--ref_fsdp_param_offload", type=bool, default=True, help="Ref FSDP param offload")

    # Critic arguments
    parser.add_argument("--critic_optim_lr", type=float, default=1e-5, help="Critic optimizer learning rate")
    parser.add_argument("--critic_model_use_remove_padding", type=bool, default=True, help="Use remove padding in critic model")
    parser.add_argument("--critic_model_path", type=str, default=None, help="Critic model path (if different from actor)")
    parser.add_argument("--critic_model_enable_gradient_checkpointing", type=bool, default=True, help="Enable gradient checkpointing for critic")
    parser.add_argument("--critic_ppo_micro_batch_size_per_gpu", type=int, default=32, help="Critic PPO micro batch size per GPU")
    parser.add_argument("--critic_fsdp_param_offload", type=bool, default=False, help="Critic FSDP param offload")
    parser.add_argument("--critic_fsdp_optimizer_offload", type=bool, default=False, help="Critic FSDP optimizer offload")

    # Algorithm arguments
    parser.add_argument("--algorithm_adv_estimator", type=str, default="grpo", help="Advantage estimator (e.g., grpo, gae)")
    parser.add_argument("--algorithm_gamma", type=float, default=1.0, help="Discount factor for future rewards")
    parser.add_argument("--algorithm_lam", type=float, default=1.0, help="GAE lambda parameter")
    parser.add_argument("--algorithm_norm_adv_by_std_in_grpo", type=bool, default=True, help="Normalize advantages by std in GRPO")
    parser.add_argument("--algorithm_use_kl_in_reward", type=bool, default=False, help="Use KL in reward")
    parser.add_argument("--algorithm_kl_penalty", type=str, default="kl", help="KL penalty type: kl, abs, mse, low_var_kl, full")
    parser.add_argument("--algorithm_kl_ctrl_type", type=str, default="fixed", help="KL control type: fixed or adaptive")
    parser.add_argument("--algorithm_kl_ctrl_kl_coef", type=float, default=0.001, help="KL coefficient")
    parser.add_argument("--algorithm_kl_ctrl_horizon", type=int, default=10000, help="Horizon for adaptive KL controller")
    parser.add_argument("--algorithm_kl_ctrl_target_kl", type=float, default=0.1, help="Target KL for adaptive controller")

    # Actor Training Parameters
    parser.add_argument("--actor_clip_ratio", type=float, default=0.2, help="PPO clip ratio")
    parser.add_argument("--actor_clip_ratio_low", type=float, default=0.2, help="Lower bound for asymmetric clipping")
    parser.add_argument("--actor_clip_ratio_high", type=float, default=0.2, help="Upper bound for asymmetric clipping")
    parser.add_argument("--actor_clip_ratio_c", type=float, default=3.0, help="Dual-clip PPO constant")
    parser.add_argument("--actor_loss_agg_mode", type=str, default="token-mean", help="Loss aggregation mode")
    parser.add_argument("--actor_ppo_epochs", type=int, default=1, help="Number of PPO epochs per batch")
    parser.add_argument("--actor_shuffle", type=bool, default=False, help="Shuffle training data across PPO epochs")
    parser.add_argument("--actor_use_dynamic_bsz", type=bool, default=False, help="Auto-adjust batch size at runtime")
    parser.add_argument("--actor_ppo_max_token_len_per_gpu", type=int, default=16384, help="Max tokens per GPU in one PPO batch")
    parser.add_argument("--actor_use_torch_compile", type=bool, default=True, help="Use torch.compile()")
    parser.add_argument("--actor_grad_clip", type=float, default=1.0, help="Gradient clipping value")
    parser.add_argument("--actor_policy_loss_mode", type=str, default="vanilla", help="Policy loss mode: vanilla, clip-cov, kl-cov, gpg")

    # Actor Optimizer Parameters
    parser.add_argument("--actor_optim_lr_warmup_steps", type=int, default=-1, help="Warmup steps")
    parser.add_argument("--actor_optim_lr_warmup_steps_ratio", type=float, default=0.0, help="Warmup steps ratio")
    parser.add_argument("--actor_optim_min_lr_ratio", type=float, default=0.0, help="Min LR ratio for cosine schedule")
    parser.add_argument("--actor_optim_num_cycles", type=float, default=0.5, help="Number of cosine cycles")
    parser.add_argument("--actor_optim_warmup_style", type=str, default="constant", help="Warmup style: constant or cosine")
    parser.add_argument("--actor_optim_weight_decay", type=float, default=0.01, help="Weight decay")

    # Actor FSDP Configuration
    parser.add_argument("--actor_fsdp_wrap_policy_min_num_params", type=int, default=0, help="Min params to trigger FSDP wrapping")
    parser.add_argument("--actor_fsdp_reshard_after_forward", type=bool, default=True, help="Reshard after forward")
    parser.add_argument("--actor_fsdp_size", type=int, default=-1, help="GPUs in each FSDP shard group")
    parser.add_argument("--actor_fsdp_forward_prefetch", type=bool, default=False, help="FSDP1 forward prefetch")
    parser.add_argument("--actor_ulysses_sequence_parallel_size", type=int, default=1, help="Sequence parallelism size")

    # Model Configuration
    parser.add_argument("--actor_model_custom_chat_template", type=str, default=None, help="Custom chat template")
    parser.add_argument("--actor_model_use_shm", type=bool, default=False, help="Use shared memory for loading")
    parser.add_argument("--actor_model_external_lib", type=str, default=None, help="External Python packages for custom models")
    parser.add_argument("--actor_model_enable_activation_offload", type=bool, default=False, help="Enable activation offloading")
    parser.add_argument("--actor_model_use_liger", type=bool, default=False, help="Use Liger for linear fusion")
    parser.add_argument("--actor_model_use_fused_kernels", type=bool, default=False, help="Use custom fused kernels")
    parser.add_argument("--actor_model_trust_remote_code", type=bool, default=False, help="Trust remote code")

    # Rollout Generation Parameters
    parser.add_argument("--rollout_mode", type=str, default="sync", help="Rollout mode: sync or async")
    parser.add_argument("--rollout_temperature", type=float, default=1.0, help="Sampling temperature")
    parser.add_argument("--rollout_top_k", type=int, default=-1, help="Top-k sampling")
    parser.add_argument("--rollout_top_p", type=float, default=1.0, help="Top-p sampling")
    parser.add_argument("--rollout_ignore_eos", type=bool, default=False, help="Ignore EOS and continue")
    parser.add_argument("--rollout_max_num_batched_tokens", type=int, default=8192, help="Max tokens in a batch")
    parser.add_argument("--rollout_max_model_len", type=int, default=None, help="Max length for rollout")
    parser.add_argument("--rollout_max_num_seqs", type=int, default=1024, help="Max sequences")
    parser.add_argument("--rollout_disable_log_stats", type=bool, default=True, help="Disable logging statistics")
    parser.add_argument("--rollout_do_sample", type=bool, default=True, help="Sample during rollout")
    parser.add_argument("--rollout_load_format", type=str, default="dummy_dtensor", help="Model weight loader")
    parser.add_argument("--rollout_layered_summon", type=bool, default=False, help="Layered summon for huge models")

    # Rollout Validation Parameters
    parser.add_argument("--rollout_val_top_k", type=int, default=-1, help="Validation top-k")
    parser.add_argument("--rollout_val_top_p", type=float, default=1.0, help="Validation top-p")
    parser.add_argument("--rollout_val_temperature", type=float, default=0, help="Validation temperature")
    parser.add_argument("--rollout_val_n", type=int, default=1, help="Validation repeat times")
    parser.add_argument("--rollout_val_do_sample", type=bool, default=False, help="Validation do_sample")

    # Reference Model Parameters
    parser.add_argument("--ref_fsdp_reshard_after_forward", type=bool, default=True, help="Ref reshard after forward")
    parser.add_argument("--ref_fsdp_forward_prefetch", type=bool, default=False, help="Ref FSDP1 forward prefetch")

    # Data Loading Parameters
    parser.add_argument("--data_tokenizer", type=str, default=None, help="Tokenizer class or path")
    parser.add_argument("--data_use_shm", type=bool, default=False, help="Use shared memory for data")
    parser.add_argument("--data_prompt_key", type=str, default="prompt", help="Field for prompt in dataset")
    parser.add_argument("--data_reward_fn_key", type=str, default="data_source", help="Field for reward function selection")
    parser.add_argument("--data_val_batch_size", type=int, default=None, help="Validation batch size")
    parser.add_argument("--data_shuffle", type=bool, default=True, help="Shuffle training data")
    parser.add_argument("--data_dataloader_num_workers", type=int, default=8, help="Dataloader workers")
    parser.add_argument("--data_validation_shuffle", type=bool, default=False, help="Shuffle validation set")
    parser.add_argument("--data_video_key", type=str, default="videos", help="Video field in multimodal dataset")
    parser.add_argument("--data_trust_remote_code", type=bool, default=False, help="Trust remote tokenizer code")

    # Trainer arguments
    parser.add_argument("--trainer_balance_batch", type=bool, default=True, help="Balance batch sizes across workers")
    parser.add_argument("--trainer_critic_warmup", type=int, default=0, help="Critic warmup")
    parser.add_argument("--trainer_logger", type=str, default='["console","wandb"]', help="Logger")
    parser.add_argument("--trainer_log_val_generations", type=int, default=0, help="Number of validation generations to log")
    parser.add_argument("--trainer_project_name", type=str, default="verl_grpo_example_geo3k", help="Project name")
    parser.add_argument("--trainer_experiment_name", type=str, default="qwen2_5_vl_7b_function_rm", help="Experiment name")
    parser.add_argument("--trainer_n_gpus_per_node", type=int, default=8, help="Number of GPUs per node")
    parser.add_argument("--trainer_nnodes", type=int, default=1, help="Number of nodes")
    parser.add_argument("--trainer_save_freq", type=int, default=20, help="Save frequency")
    parser.add_argument("--trainer_save_contents", type=str, default='["model","optimizer","extra","hf_model"]', help="Save Contents")
    parser.add_argument("--trainer_test_freq", type=int, default=5, help="Test frequency")
    parser.add_argument("--trainer_total_epochs", type=int, default=15, help="Total epochs")
    parser.add_argument("--trainer_resume_mode", type=str, default="auto", help="Resume mode: auto, disable, or resume_path")
    parser.add_argument("--trainer_resume_from_path", type=str, default=None, help="Path to resume from")
    parser.add_argument("--trainer_val_before_train", type=bool, default=True, help="Run validation before training")
    parser.add_argument("--trainer_val_only", type=bool, default=False, help="Run validation only")
    parser.add_argument("--trainer_device", type=str, default="cuda", help="Device: cuda or cpu")
    parser.add_argument("--trainer_use_legacy_worker_impl", type=str, default="auto", help="Use legacy worker: auto, enable, disable")
    parser.add_argument("--total_training_steps", type=int, help="Total number of training steps")

    # Checkpoint arguments
    parser.add_argument("--actor_checkpoint_load_contents", type=str, default=None, help="What to load from checkpoint")

    # Output arguments
    parser.add_argument("--output_dir", type=str, default="./output", help="Output directory for model artifacts")

    return parser.parse_args()


def _log_user_error(message: str):
    """Log a user error message.

    Args:
        message (str): The error message to log.
    """
    raise ACFTValidationException._with_error(
        AzureMLError.create(
            ACFTUserError,
            pii_safe_message=(
                message
            )
        )
    )


def bool_to_str(value):
    """Convert boolean to string for command line."""
    return str(value).lower() if isinstance(value, bool) else str(value)


def quote_if_needed(value):
    """Quote a value if it contains spaces or special characters.

    Args:
        value: The value to potentially quote

    Returns:
        str: The quoted or unquoted value as appropriate
    """
    import shlex
    value_str = str(value)
    # If the value contains spaces, parentheses, or other special chars, quote it
    if ' ' in value_str or '(' in value_str or ')' in value_str or any(c in value_str for c in ['&', '|', ';', '<', '>', '$', '`', '"', "'"]):
        return shlex.quote(value_str)
    return value_str


def find_model_config_path(base_path):
    """Find the subdirectory containing config.json file.

    Args:
        base_path (str): The base path to search for config.json

    Returns:
        str: The path containing config.json, or the original path if config.json is found at base level
    """
    import os

    # Check if config.json exists in the base path
    if os.path.exists(os.path.join(base_path, 'config.json')):
        logger.info(f"config.json found at base path: {base_path}")
        return base_path

    # If not, enumerate all subdirectories to find config.json
    logger.info(f"Searching for config.json in subdirectories of {base_path}")

    if not os.path.exists(base_path):
        logger.warning(f"Base path does not exist: {base_path}")
        return base_path

    try:
        # List all items in the base path
        for item in os.listdir(base_path):
            item_path = os.path.join(base_path, item)

            # Check if it's a directory
            if os.path.isdir(item_path):
                config_path = os.path.join(item_path, 'config.json')
                if os.path.exists(config_path):
                    logger.info(f"config.json found in subdirectory: {item_path}")
                    return item_path

        logger.warning(f"config.json not found in {base_path} or its subdirectories")
        return base_path

    except Exception as e:
        logger.error(f"Error while searching for config.json: {e}")
        return base_path


def get_current_ip():
    """Get the current machine's IP address."""
    try:
        # Connect to a remote address to find the local IP
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception as e:
        logger.warning(f"Could not determine IP address: {e}")
        return None


def setup_ray_cluster():
    """Setup Ray cluster based on distributed configuration using agent mode."""
    RAY_PORT = 20001  # Changed from 18125 to avoid conflict with worker ports (10002-19999)

    # Parse the AZUREML_CR_DISTRIBUTED_CONFIG environment variable
    distributed_config = os.environ.get('AZUREML_CR_DISTRIBUTED_CONFIG')
    if not distributed_config:
        logger.info("No distributed config found, running in single node mode")
        return True  # Allow execution on single node

    try:
        config = json.loads(distributed_config)
        host_list = config.get('host_list', [])

        if not host_list:
            logger.info("No host list found in config, running in single node mode")
            return True

        # Get current IP
        current_ip = get_current_ip()
        if not current_ip:
            logger.error("Could not determine current IP address")
            return False

        # Check if current IP is the first in the list (head node)
        if host_list[0] == current_ip:
            logger.info("This node is the Ray head node")
            try:
                # Start Ray head node using subprocess with agent mode and avoid port conflicts
                import subprocess
                result = subprocess.run([
                    "ray", "start", "--head",
                    f"--port={RAY_PORT}",
                    "--disable-usage-stats",
                    "--include-dashboard=false",  # Disable dashboard to avoid additional port conflicts
                ], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

                logger.info(f"STDOUT:\n{result.stdout}")
                if result.stderr:
                    logger.error(f"STDERR:\n{result.stderr}")

                if result.returncode != 0:
                    logger.error(f"Ray head node failed to start with return code {result.returncode}")
                    return False

                logger.info(f"Ray head node initialized on port {RAY_PORT}")
                return True
            except Exception as e:
                logger.error(f"Failed to initialize Ray head node: {e}")
                return False
        else:
            # This is a worker node, connect to head node using agent mode
            head_ip = host_list[0]
            logger.info(f"This node is a Ray worker, connecting to head at {head_ip}:{RAY_PORT}")

            try:
                import subprocess
                RETRY_COUNT = 5
                for retry in range(RETRY_COUNT):
                    result = subprocess.run([
                        "ray", "start",
                        f"--address={head_ip}:{RAY_PORT}",
                        "--disable-usage-stats",
                    ], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

                    logger.info(f"STDOUT:\n{result.stdout}")
                    if result.stderr:
                        logger.error(f"STDERR:\n{result.stderr}")

                    if result.returncode != 0:
                        logger.error(f"Ray worker failed to connect with return code {result.returncode}")
                        if retry < RETRY_COUNT - 1:
                            logger.info(f"Retrying connection to head node ({retry + 1}/{RETRY_COUNT})...")
                            time.sleep(5*(retry + 1))  # Exponential backoff
                    else:
                        break
                if result.returncode != 0:
                    logger.error("Exceeded maximum retries to connect to Ray head node")
                    return False
                logger.info("Successfully connected Ray worker to head node")
                # Check Ray cluster status after successful connection
                try:
                    for t in range(3):
                        status_result = subprocess.run([
                            "ray", "status"
                        ], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

                        logger.info("Ray status output:")
                        logger.info(f"Ray status STDOUT:\n {status_result.stdout}")
                        if status_result.stderr:
                            logger.error(f"Ray status STDERR:\n {status_result.stderr}")
                        time.sleep(2*t)

                except Exception as e:
                    logger.warning(f"Failed to get Ray status: {e}")
                return False  # Worker nodes should not execute main
            except Exception as e:
                logger.error(f"Failed to connect Ray worker: {e}")
                return False

    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse distributed config: {e}")
        return False
    except Exception as e:
        logger.error(f"Error setting up Ray cluster: {e}")
        return False


def main():
    """Main function to run Direct Reasoning Optimization."""
    try:
        # Setup Ray cluster and determine if this node should execute main logic
        is_head_node = setup_ray_cluster()

        # Initialize Azure ML logging
        logger = get_logger_app("azureml.acft.contrib.nlp.entry_point.run_training")

        # Set logging parameters
        set_logging_parameters(
            task_type="chat-completion",
            acft_custom_dimensions={
                LoggingLiterals.PROJECT_NAME: "Verl_Trainer",
                LoggingLiterals.PROJECT_VERSION_NUMBER: "1.0.0",
                LoggingLiterals.COMPONENT_NAME: "verl_component"
            }
        )

        logger.info("Starting Direct Reasoning Optimization component")

        # Parse arguments
        args = parse_args()

        # Install PyPI package overrides if provided
        if args.pypi_packages_override:
            logger.info(f"Installing PyPI package overrides: {args.pypi_packages_override}")
            packages = [pkg.strip() for pkg in args.pypi_packages_override.split(',') if pkg.strip()]
            for package in packages:
                logger.info(f"Installing package: {package}")
                try:
                    result = subprocess.run(
                        [sys.executable, "-m", "pip", "install", "--upgrade", package],
                        check=True,
                        capture_output=True,
                        text=True
                    )
                    logger.info(f"Successfully installed {package}")
                    if result.stdout:
                        logger.info(f"STDOUT: {result.stdout}")
                except subprocess.CalledProcessError as e:
                    logger.error(f"Failed to install {package}: {e}")
                    if e.stdout:
                        logger.error(f"STDOUT: {e.stdout}")
                    if e.stderr:
                        logger.error(f"STDERR: {e.stderr}")
                    logger.warning(f"Continuing with remaining packages despite failure for: {package}")
            logger.info("Completed PyPI package overrides installation")

        logger.info("Arguments parsed successfully")
        logger.info(f"Engine type: {args.ENGINE}")
        logger.info(f"Training data: {args.data_train_files}")
        logger.info(f"Validation data: {args.data_val_files}")
        logger.info(f"Output directory: {args.output_dir}")

        # Find the correct actor model path containing config.json
        original_actor_model_path = args.actor_model_path
        args.actor_model_path = find_model_config_path(args.actor_model_path)
        if original_actor_model_path != args.actor_model_path:
            logger.info(f"Updated actor model path from {original_actor_model_path} to {args.actor_model_path}")

        # Find the correct critic model path containing config.json if provided
        if args.critic_model_path:
            original_critic_model_path = args.critic_model_path
            args.critic_model_path = find_model_config_path(args.critic_model_path)
            if original_critic_model_path != args.critic_model_path:
                logger.info(f"Updated critic model path from {original_critic_model_path} to {args.critic_model_path}")

        # Create output directory if it doesn't exist
        os.makedirs(args.output_dir, exist_ok=True)

        # Set debug mode
        os.environ["PYTHONPATH"] = os.environ.get("PYTHONPATH", "") + ":" + os.getcwd()
        # Build the command
        FALSE=False
        TRUE=True
        cmd = [
            "python3", "-m", "verl.trainer.main_ppo",
            f"algorithm.adv_estimator={args.algorithm_adv_estimator}",
            f"data.train_files={quote_if_needed(args.data_train_files)}",
            f"data.val_files={quote_if_needed(args.data_val_files)}",
            f"data.train_batch_size={args.data_train_batch_size}",
            f"data.max_prompt_length={args.data_max_prompt_length}",
            f"data.max_response_length={args.data_max_response_length}",
            f"data.filter_overlong_prompts={bool_to_str(args.data_filter_overlong_prompts)}",
            f"data.truncation={args.data_truncation}",
            f"data.image_key={args.data_image_key}",
            f"actor_rollout_ref.model.path={quote_if_needed(args.actor_model_path)}",
            f"actor_rollout_ref.actor.optim.lr={args.actor_optim_lr}",
            f"actor_rollout_ref.model.use_remove_padding={bool_to_str(args.actor_model_use_remove_padding)}",
            f"actor_rollout_ref.actor.ppo_mini_batch_size={args.actor_ppo_mini_batch_size}",
            f"actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu={args.actor_ppo_micro_batch_size_per_gpu}",
            f"actor_rollout_ref.model.lora_rank={args.actor_model_lora_rank}",
            f"actor_rollout_ref.model.lora_alpha={args.actor_model_lora_alpha}",
            f"actor_rollout_ref.model.target_modules={args.actor_model_target_modules}",
            f"actor_rollout_ref.model.exclude_modules={args.actor_model_exclude_modules}",
            f"actor_rollout_ref.actor.use_kl_loss={bool_to_str(args.actor_use_kl_loss)}",
            f"actor_rollout_ref.actor.kl_loss_coef={args.actor_kl_loss_coef}",
            f"actor_rollout_ref.actor.kl_loss_type={args.actor_kl_loss_type}",
            f"actor_rollout_ref.actor.entropy_coeff={args.actor_entropy_coeff}",
            f"actor_rollout_ref.model.enable_gradient_checkpointing=" +
            f"{bool_to_str(args.actor_model_enable_gradient_checkpointing)}",
            f"actor_rollout_ref.actor.fsdp_config.param_offload={bool_to_str(args.actor_fsdp_param_offload)}",
            f"actor_rollout_ref.actor.fsdp_config.optimizer_offload={bool_to_str(args.actor_fsdp_optimizer_offload)}",
            f"actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=" +
            f"{args.rollout_log_prob_micro_batch_size_per_gpu}",
            f"actor_rollout_ref.rollout.tensor_model_parallel_size={args.rollout_tensor_model_parallel_size}",
            f"actor_rollout_ref.rollout.name={args.ENGINE}",
            f"actor_rollout_ref.rollout.engine_kwargs.vllm.disable_mm_preprocessor_cache=" +
            f"{bool_to_str(args.rollout_disable_mm_preprocessor_cache)}",
            f"actor_rollout_ref.rollout.gpu_memory_utilization={args.rollout_gpu_memory_utilization}",
            f"actor_rollout_ref.rollout.enable_chunked_prefill={bool_to_str(FALSE)}",
            f"actor_rollout_ref.rollout.enforce_eager={bool_to_str(args.rollout_enforce_eager)}",
            f"actor_rollout_ref.rollout.free_cache_engine={bool_to_str(args.rollout_free_cache_engine)}",
            f"actor_rollout_ref.rollout.n={args.rollout_n}",
            f"actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu={args.ref_log_prob_micro_batch_size_per_gpu}",
            f"actor_rollout_ref.ref.fsdp_config.param_offload={bool_to_str(args.ref_fsdp_param_offload)}",
            f"actor_rollout_ref.ref.fsdp_config.reshard_after_forward={bool_to_str(args.ref_fsdp_reshard_after_forward)}",
            f"algorithm.use_kl_in_reward={bool_to_str(args.algorithm_use_kl_in_reward)}",
            f"algorithm.gamma={args.algorithm_gamma}",
            f"algorithm.lam={args.algorithm_lam}",
            f"algorithm.norm_adv_by_std_in_grpo={bool_to_str(args.algorithm_norm_adv_by_std_in_grpo)}",
            f"algorithm.kl_penalty={args.algorithm_kl_penalty}",
            f"algorithm.kl_ctrl.type={args.algorithm_kl_ctrl_type}",
            f"algorithm.kl_ctrl.kl_coef={args.algorithm_kl_ctrl_kl_coef}",
            f"algorithm.kl_ctrl.horizon={args.algorithm_kl_ctrl_horizon}",
            f"algorithm.kl_ctrl.target_kl={args.algorithm_kl_ctrl_target_kl}",
            f"actor_rollout_ref.actor.clip_ratio={args.actor_clip_ratio}",
            f"actor_rollout_ref.actor.clip_ratio_low={args.actor_clip_ratio_low}",
            f"actor_rollout_ref.actor.clip_ratio_high={args.actor_clip_ratio_high}",
            f"actor_rollout_ref.actor.clip_ratio_c={args.actor_clip_ratio_c}",
            f"actor_rollout_ref.actor.loss_agg_mode={args.actor_loss_agg_mode}",
            f"actor_rollout_ref.actor.ppo_epochs={args.actor_ppo_epochs}",
            f"actor_rollout_ref.actor.shuffle={bool_to_str(args.actor_shuffle)}",
            f"actor_rollout_ref.actor.use_dynamic_bsz={bool_to_str(args.actor_use_dynamic_bsz)}",
            f"actor_rollout_ref.actor.ppo_max_token_len_per_gpu={args.actor_ppo_max_token_len_per_gpu}",
            f"actor_rollout_ref.actor.use_torch_compile={bool_to_str(args.actor_use_torch_compile)}",
            f"actor_rollout_ref.actor.grad_clip={args.actor_grad_clip}",
            f"actor_rollout_ref.actor.policy_loss.loss_mode={args.actor_policy_loss_mode}",
            f"actor_rollout_ref.actor.optim.lr_warmup_steps={args.actor_optim_lr_warmup_steps}",
            f"actor_rollout_ref.actor.optim.lr_warmup_steps_ratio={args.actor_optim_lr_warmup_steps_ratio}",
            f"actor_rollout_ref.actor.optim.min_lr_ratio={args.actor_optim_min_lr_ratio}",
            f"actor_rollout_ref.actor.optim.num_cycles={args.actor_optim_num_cycles}",
            f"actor_rollout_ref.actor.optim.warmup_style={args.actor_optim_warmup_style}",
            f"actor_rollout_ref.actor.optim.weight_decay={args.actor_optim_weight_decay}",
            f"actor_rollout_ref.actor.fsdp_config.wrap_policy.min_num_params={args.actor_fsdp_wrap_policy_min_num_params}",
            f"actor_rollout_ref.actor.fsdp_config.reshard_after_forward={bool_to_str(args.actor_fsdp_reshard_after_forward)}",
            f"actor_rollout_ref.actor.fsdp_config.fsdp_size={args.actor_fsdp_size}",
            f"actor_rollout_ref.actor.fsdp_config.forward_prefetch={bool_to_str(args.actor_fsdp_forward_prefetch)}",
            f"actor_rollout_ref.actor.ulysses_sequence_parallel_size={args.actor_ulysses_sequence_parallel_size}",
            f"actor_rollout_ref.rollout.mode={args.rollout_mode}",
            f"actor_rollout_ref.rollout.temperature={args.rollout_temperature}",
            f"actor_rollout_ref.rollout.top_k={args.rollout_top_k}",
            f"actor_rollout_ref.rollout.top_p={args.rollout_top_p}",
            f"actor_rollout_ref.rollout.ignore_eos={bool_to_str(args.rollout_ignore_eos)}",
            f"actor_rollout_ref.rollout.max_num_batched_tokens={args.rollout_max_num_batched_tokens}",
            f"actor_rollout_ref.rollout.max_num_seqs={args.rollout_max_num_seqs}",
            f"actor_rollout_ref.rollout.disable_log_stats={bool_to_str(args.rollout_disable_log_stats)}",
            f"actor_rollout_ref.rollout.do_sample={bool_to_str(args.rollout_do_sample)}",
            f"actor_rollout_ref.rollout.load_format={args.rollout_load_format}",
            f"actor_rollout_ref.rollout.layered_summon={bool_to_str(args.rollout_layered_summon)}",
            f"actor_rollout_ref.rollout.val_kwargs.top_k={args.rollout_val_top_k}",
            f"actor_rollout_ref.rollout.val_kwargs.top_p={args.rollout_val_top_p}",
            f"actor_rollout_ref.rollout.val_kwargs.temperature={args.rollout_val_temperature}",
            f"actor_rollout_ref.rollout.val_kwargs.n={args.rollout_val_n}",
            f"actor_rollout_ref.rollout.val_kwargs.do_sample={bool_to_str(args.rollout_val_do_sample)}",
            f"data.shuffle={bool_to_str(args.data_shuffle)}",
            f"data.dataloader_num_workers={args.data_dataloader_num_workers}",
            f"data.validation_shuffle={bool_to_str(args.data_validation_shuffle)}",
            f"data.prompt_key={args.data_prompt_key}",
            f"data.reward_fn_key={args.data_reward_fn_key}",
            f"trainer.balance_batch={bool_to_str(args.trainer_balance_batch)}",
            f"trainer.log_val_generations={args.trainer_log_val_generations}",
            f"trainer.resume_mode={args.trainer_resume_mode}",
            f"trainer.val_before_train={bool_to_str(args.trainer_val_before_train)}",
            f"trainer.val_only={bool_to_str(args.trainer_val_only)}",
            f"trainer.device={args.trainer_device}",
            f"trainer.use_legacy_worker_impl={args.trainer_use_legacy_worker_impl}",
            f"trainer.critic_warmup={args.trainer_critic_warmup}",
            f"trainer.logger={args.trainer_logger}",
            f"trainer.project_name={args.trainer_project_name}",
            f"trainer.experiment_name={args.trainer_experiment_name}",
            f"trainer.n_gpus_per_node={args.trainer_n_gpus_per_node}",
            f"trainer.nnodes={args.trainer_nnodes}",
            f"trainer.save_freq={args.trainer_save_freq}",
            f"trainer.test_freq={args.trainer_test_freq}",
            f"trainer.total_epochs={args.trainer_total_epochs}",
            f"actor_rollout_ref.actor.checkpoint.save_contents={args.trainer_save_contents}",
            f"actor_rollout_ref.rollout.dtype={args.rollout_dtype}",
            f"actor_rollout_ref.actor.strategy={args.actor_strategy}",
            f"actor_rollout_ref.actor.fsdp_config.offload_policy={bool_to_str(args.actor_fsdp_config_offload_policy)}"
        ]
        
        # Add FSDP mixed precision config if specified (these may not be in base config)
        if hasattr(args, 'actor_fsdp_config_mixed_precision_param_dtype') and args.actor_fsdp_config_mixed_precision_param_dtype:
            cmd.append(f"+actor_rollout_ref.actor.fsdp_config.mixed_precision.param_dtype={args.actor_fsdp_config_mixed_precision_param_dtype}")
        if hasattr(args, 'actor_fsdp_config_mixed_precision_reduce_dtype') and args.actor_fsdp_config_mixed_precision_reduce_dtype:
            cmd.append(f"+actor_rollout_ref.actor.fsdp_config.mixed_precision.reduce_dtype={args.actor_fsdp_config_mixed_precision_reduce_dtype}")
        if hasattr(args, 'actor_fsdp_config_mixed_precision_buffer_dtype') and args.actor_fsdp_config_mixed_precision_buffer_dtype:
            cmd.append(f"+actor_rollout_ref.actor.fsdp_config.mixed_precision.buffer_dtype={args.actor_fsdp_config_mixed_precision_buffer_dtype}")

        # Add critic parameters
        cmd.append(f"critic.optim.lr={args.critic_optim_lr}")
        cmd.append(f"critic.model.use_remove_padding={bool_to_str(args.critic_model_use_remove_padding)}")
        if args.critic_model_path:
            cmd.append(f"critic.model.path={quote_if_needed(args.critic_model_path)}")
        cmd.append(f"critic.model.enable_gradient_checkpointing={bool_to_str(args.critic_model_enable_gradient_checkpointing)}")
        cmd.append(f"critic.ppo_micro_batch_size_per_gpu={args.critic_ppo_micro_batch_size_per_gpu}")
        cmd.append(f"critic.model.fsdp_config.param_offload={bool_to_str(args.critic_fsdp_param_offload)}")
        cmd.append(f"critic.model.fsdp_config.optimizer_offload={bool_to_str(args.critic_fsdp_optimizer_offload)}")
        #   f"actor_rollout_ref.rollout.layered_summon={bool_to_str(TRUE)}",
        #   f"actor_rollout_ref.rollout.load_format=safetensors",
        # Add total_training_steps if provided
        if args.total_training_steps is not None:
            cmd.append(f"trainer.total_training_steps={args.total_training_steps}")

        # Add optional parameters if provided
        if args.rollout_max_model_len is not None:
            cmd.append(f"actor_rollout_ref.rollout.max_model_len={args.rollout_max_model_len}")

        if args.data_val_batch_size is not None:
            cmd.append(f"data.val_batch_size={args.data_val_batch_size}")

        if args.data_tokenizer is not None:
            cmd.append(f"data.tokenizer={quote_if_needed(args.data_tokenizer)}")

        if args.actor_model_custom_chat_template is not None:
            cmd.append(f"actor_rollout_ref.model.custom_chat_template={quote_if_needed(args.actor_model_custom_chat_template)}")

        if args.actor_model_external_lib is not None:
            cmd.append(f"actor_rollout_ref.model.external_lib={quote_if_needed(args.actor_model_external_lib)}")

        if args.actor_checkpoint_load_contents is not None:
            cmd.append(f"actor_rollout_ref.actor.checkpoint.load_contents={args.actor_checkpoint_load_contents}")

        if args.trainer_resume_from_path is not None:
            cmd.append(f"trainer.resume_from_path={quote_if_needed(args.trainer_resume_from_path)}")


        # Add model configuration flags
        cmd.extend([
            f"actor_rollout_ref.model.use_shm={bool_to_str(args.actor_model_use_shm)}",
            f"actor_rollout_ref.model.enable_activation_offload={bool_to_str(args.actor_model_enable_activation_offload)}",
            f"actor_rollout_ref.model.use_liger={bool_to_str(args.actor_model_use_liger)}",
            f"actor_rollout_ref.model.use_fused_kernels={bool_to_str(args.actor_model_use_fused_kernels)}",
            f"actor_rollout_ref.model.trust_remote_code={bool_to_str(args.actor_model_trust_remote_code)}",
            f"data.use_shm={bool_to_str(args.data_use_shm)}",
            f"data.video_key={args.data_video_key}",
            f"data.trust_remote_code={bool_to_str(args.data_trust_remote_code)}",
        ])

        # Add remaining parameters
        cmd.extend([
            f"trainer.default_local_dir={quote_if_needed(args.output_dir)}",
            f"trainer.project_name={args.trainer_project_name}",
        ])

        # Add custom dataset class configuration to support JSONL files
        # Get the directory where this script is located
        script_dir = os.path.dirname(os.path.abspath(__file__))
        jsonl_dataset_path = os.path.join(script_dir, "jsonl_dataset.py")

        logger.info(f"Using custom JSONL dataset class from: {jsonl_dataset_path}")

        # Add custom dataset class parameters to the command
        cmd.extend([
            f"data.custom_cls.path={quote_if_needed(jsonl_dataset_path)}",
            "data.custom_cls.name=JSONLDataset",
        ])

        logger.info("Executing verl.trainer.main_ppo command with custom JSONL dataset")
        logger.info(f"Command: {' '.join(cmd)}")

        # Execute the command
        if not is_head_node:
            logger.info("This is a worker node, skipping main execution")
            return 0  # Worker nodes should not execute main

        logger.info(f"Running command : {cmd}")
        result = subprocess.run(cmd, check=False, capture_output=True, text=True)

        if result.stdout:
            logger.info(f"STDOUT:\n{result.stdout}")
        if result.stderr:
            logger.error(f"STDERR:\n{result.stderr}")

        if result.returncode != 0:
            error_msg = f"Command failed with return code {result.returncode}"
            logger.error(error_msg)
            logger.error(f"Command: {' '.join(cmd)}")
            _log_user_error(error_msg)
            sys.exit(result.returncode)

        logger.info("Command executed successfully")
        logger.info(f"Return code: {result.returncode}")

        return result.returncode

    except subprocess.CalledProcessError as e:
        error_msg = f"Command failed with return code {e.returncode}"
        if hasattr(e, 'stdout') and e.stdout:
            logger.error(f"STDOUT:\n{e.stdout}")
        if hasattr(e, 'stderr') and e.stderr:
            logger.error(f"STDERR:\n{e.stderr}")
        _log_user_error(error_msg)
        sys.exit(e.returncode)

    except Exception as e:
        error_msg = f"Unexpected error occurred: {str(e)}"
        _log_user_error(error_msg)
        sys.exit(1)


if __name__ == "__main__":
    main()
