import argparse
import hashlib
import logging
import math
import os
import time
from collections import defaultdict

import torch
import torch.distributed as dist
from accelerate.utils import set_seed
from datasets import load_dataset
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import MixedPrecision, ShardingStrategy, StateDictType
from tqdm import tqdm
from transformers import AutoConfig, AutoModelForCausalLM, AutoProcessor, AutoTokenizer

from specforge import (
    AutoDistributedTargetModel,
    AutoDraftModelConfig,
    AutoEagle3DraftModel,
    OnlineEagle3Model,
    QwenVLOnlineEagle3Model,
)
from specforge.data import (
    build_eagle3_dataset,
    generate_vocab_mapping_file,
    prepare_dp_dataloaders,
)
from specforge.distributed import (
    destroy_distributed,
    get_dp_group,
    get_tp_device_mesh,
    init_distributed,
)
from specforge.optimizer import BF16Optimizer
from specforge.tracker import create_tracker, get_tracker_class, Tracker
from specforge.utils import (
    create_draft_config_from_target,
    get_last_checkpoint,
    print_on_rank0,
    print_with_rank,
    rank_0_priority,
)

# Azure ML logging
from azureml.acft.common_components import get_logger_app, set_logging_parameters, LoggingLiterals
from azureml.acft.common_components.utils.logging_utils import SystemSettings

# Initialize logger
logger = get_logger_app("azureml.acft.contrib.hf.scripts.src.eagle3.train_eagle3")

COMPONENT_NAME = "ACFT-Eagle3-Training"
PROJECT_NAME = "azureml-acft-hf-nlp"
VERSION = "1.0.0"


class AzureMLTrackerAdapter:
    """
    AzureML tracker adapter for Eagle3 training.
    Integrates with Azure Machine Learning Run context for metrics logging.
    Based on AzureMLLogger implementation from aml_log_tracking.py.
    """

    def __init__(self, args, output_dir: str):
        self.rank = dist.get_rank() if dist.is_initialized() else 0
        self.output_dir = output_dir
        self.azureml_run = None
        self.logged_steps = []
        self.log_metrics_at_root = True
        self.set_log_prefix = True

        if self.rank == 0:
            # Initialize AzureML run context
            try:
                from azureml.core.run import Run
                self.azureml_run = Run.get_context()

                # Check if running offline (local development)
                if self.azureml_run and "OfflineRun" in self.azureml_run.id:
                    print("Warning: Running in offline mode - AzureML logging disabled")
                    logger.warning("Running in offline mode - AzureML logging disabled")
                    self.azureml_run = None
                else:
                    print(f"AzureML logging initialized for Eagle3 training")
                    logger.info("AzureML logging initialized for Eagle3 training")

                    # Log configuration as run properties if available
                    if args:
                        try:
                            # Convert args to flat dict for run properties
                            flat_config = self._flatten_config(vars(args))
                            for key, value in flat_config.items():
                                # AzureML run properties have string length limits
                                if isinstance(value, str) and len(value) < 1000:
                                    self.azureml_run.tag(key, value)
                        except Exception as e:
                            print(f"Warning: Failed to log config as run properties: {e}")
                            logger.warning(f"Failed to log config as run properties: {e}")

            except ImportError:
                print("Warning: azureml-core not available - AzureML logging disabled")
                logger.warning("azureml-core not available - AzureML logging disabled")
                self.azureml_run = None
            except Exception as e:
                print(f"Warning: Failed to initialize AzureML run context: {e}")
                logger.warning(f"Failed to initialize AzureML run context: {e}")
                self.azureml_run = None

    def _flatten_config(self, config, parent_key='', sep='_'):
        """Flatten nested configuration dictionary"""
        items = []
        if isinstance(config, dict):
            for k, v in config.items():
                new_key = f"{parent_key}{sep}{k}" if parent_key else k
                if isinstance(v, dict):
                    items.extend(self._flatten_config(v, new_key, sep=sep).items())
                else:
                    items.append((new_key, str(v)))
        return dict(items)

    def _should_log_to_parent(self):
        """Check if we should log to parent pipeline run"""
        if not self.azureml_run:
            return None

        try:
            parent_run = self.azureml_run.parent
            child_run = None

            # Navigate up the hierarchy to find pipeline run
            while parent_run is not None and (
                parent_run.type == "PipelineRun" or
                parent_run.type == "StepRun" or
                parent_run.type.lower() == "finetunerun"
            ):
                child_run = parent_run
                parent_run = parent_run.parent
            return child_run
        except Exception:
            return None

    def log(self, log_dict, step=None):
        """Log metrics to AzureML run"""
        if self.rank != 0 or not self.azureml_run:
            return

        try:
            from math import isnan

            for k, v in log_dict.items():
                if isinstance(v, (int, float)) and not isnan(v) and (step, k) not in self.logged_steps:

                    # Handle prefix modification if needed
                    if not self.set_log_prefix:
                        eval_prefix = 'eval_'
                        train_prefix = 'train_'
                        if k.startswith(eval_prefix):
                            k = k[len(eval_prefix):]
                        if k.startswith(train_prefix):
                            k = k[len(train_prefix):]
                            k = k + '_train'

                    # Log to current run
                    self.azureml_run.log(k, v, description=k, step=step)

                    # Log to parent pipeline run if enabled
                    if self.log_metrics_at_root:
                        parent_run = self._should_log_to_parent()
                        if parent_run:
                            try:
                                parent_run.log(k, v, description=k, step=step)
                            except Exception as e:
                                print(f"Warning: Failed to log to parent run: {e}")

                    # Track logged steps to avoid duplicates
                    if step is not None:
                        self.logged_steps.append((step, k))

        except Exception as e:
            print(f"Warning: AzureML logging failed: {e}")
            logger.warning(f"AzureML logging failed: {e}")

    def close(self):
        """Cleanup method for AzureML tracker"""
        # AzureML runs are managed by the platform, no explicit cleanup needed
        if self.rank == 0:
            logger.info("AzureML tracker closed")


def find_model_config_path(root_path):
    """
    Recursively search for config.json in the root_path and its subdirectories.
    Returns the directory containing config.json if found, otherwise returns the original path.

    Args:
        root_path: The root directory to search in

    Returns:
        Path to the directory containing config.json
    """
    if not os.path.exists(root_path):
        print_on_rank0(f"Warning: Path {root_path} does not exist")
        return root_path

    # Check if config.json exists in the root path
    config_path = os.path.join(root_path, "config.json")
    if os.path.isfile(config_path):
        print_on_rank0(f"Found config.json at: {root_path}")
        return root_path

    # Search in subdirectories
    if os.path.isdir(root_path):
        for dirpath, dirnames, filenames in os.walk(root_path):
            if "config.json" in filenames:
                print_on_rank0(f"Found config.json in subdirectory: {dirpath}")
                return dirpath

    print_on_rank0(f"Warning: config.json not found in {root_path} or its subdirectories")
    return root_path


def parse_args():
    parser = argparse.ArgumentParser(description="Train Eagle3 with online data")

    # add model-related arguments
    parser.add_argument("--target_model_path", type=str, required=True)
    parser.add_argument(
        "--draft_model_config",
        type=str,
        required=False,
        help="Draft model config path. If not provided, will auto-generate from target model.",
    )
    parser.add_argument(
        "--embedding_key",
        type=str,
        default="model.embed_tokens.weight",
        help="The key of the embedding weight to load from the target model",
    )

    # add training-related arguments
    parser.add_argument("--dataset_train_split", type=str, required=False)
    parser.add_argument("--dataset_validation_split", type=str, default=None)
    parser.add_argument("--num_epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--max_length", type=int, default=2048)
    parser.add_argument("--warmup_ratio", type=float, default=0.015)
    parser.add_argument(
        "--total_steps",
        type=int,
        default=None,
        help="Total training steps. If not provided, will be calculated as num_epochs * steps_per_epoch",
    )
    parser.add_argument("--max_grad_norm", type=float, default=0.5)
    parser.add_argument(
        "--log_steps", type=int, default=50, help="Log training metrics every N steps"
    )
    parser.add_argument(
        "--ttt_length",
        type=int,
        default=7,
        help="The length for Test-Time Training (TTT).",
    )

    # data processing type
    parser.add_argument("--chat_template", type=str, default="llama3")

    # distributed training
    parser.add_argument("--tp_size", type=int, default=1)
    parser.add_argument("--dp_size", type=int, default=1)
    parser.add_argument("--draft_global_batch_size", type=int, default=8)
    parser.add_argument(
        "--draft_micro_batch_size",
        type=int,
        default=1,
        help="Micro batch size for draft model",
    )
    parser.add_argument("--draft_accumulation_steps", type=int, default=1)

    # other args
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--eval_interval", type=int, default=1)
    parser.add_argument("--save_interval", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--dist_timeout",
        type=int,
        default=20,
        help="Timeout for collective communication in minutes",
    )
    parser.add_argument("--attention_backend", type=str, default="flex_attention")

    # resume
    parser.add_argument("--resume", type=bool, default=False)
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help="Path to a checkpoint directory to resume training from. Used when resume is true and no checkpoints exist in output folder, or when resume is false to initialize from a pretrained draft model checkpoint.",
    )

    # tracking
    parser.add_argument("--report_to", type=str, default="azure_ml")

    parser.add_argument("--build_dataset_num_proc", type=int, default=8)
    parser.add_argument("--verbose", type=bool, default=False)
    parser.add_argument("--profile", type=bool, default=False)
    parser.add_argument("--profile_start_step", type=int, default=30)
    parser.add_argument("--profile_num_steps", type=int, default=4)
    parser.add_argument("--profile_record_shapes", type=bool, default=False)

    args = parser.parse_args()

    # Hardcode parameters not exposed to users
    args.is_vlm = False  # VLM support is not exposed to users
    args.is_preformatted = False  # Data should not be preformatted

    return parser, args


def main():
    # Set TORCHINDUCTOR_CACHE_DIR
    script_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.dirname(script_dir)
    os.environ['TORCHINDUCTOR_CACHE_DIR'] = os.path.join(root_dir, 'cache', 'compiled_kernels')

    # initialize
    parser, args = parse_args()

    # Appending the global rank to log file generated by each process. This is to avoid issues in multi-node runs
    # where having the same file name in each node is causing issues during file upload.
    rank = os.environ.get("RANK", "0")
    SystemSettings.LOG_FILENAME = SystemSettings.LOG_FILENAME + f'.{rank}'

    # Set logging parameters
    set_logging_parameters(
        task_type="eagle3_training",
        acft_custom_dimensions={
            LoggingLiterals.PROJECT_NAME: PROJECT_NAME,
            LoggingLiterals.PROJECT_VERSION_NUMBER: VERSION,
            LoggingLiterals.COMPONENT_NAME: COMPONENT_NAME,
        },
        azureml_pkg_denylist_logging_patterns=[],
        log_level=logging.INFO,
    )

    logger.info("Starting Eagle3 training component")
    logger.info(f"Component version: {VERSION}")

    # Hardcode cache_dir value
    cache_dir = "./cache"
    logger.info(f"Using cache directory: {cache_dir}")

    set_seed(args.seed)

    # Initialize distributed training
    try:
        # BUGFIX: Set device using LOCAL_RANK before init_distributed
        # to avoid duplicate GPU assignment issue in init_distributed()
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        num_gpus = torch.cuda.device_count()
        world_size = int(os.environ.get("WORLD_SIZE", 1))
        print(f"Distributed setup: LOCAL_RANK={local_rank}, WORLD_SIZE={world_size}, Available GPUs={num_gpus}")

        # Validate that requested GPUs don't exceed available GPUs
        if local_rank >= num_gpus:
            error_msg = (
                f"ERROR: The parameter 'number_of_gpu_to_use_eagle3_training' is set to {world_size}, "
                f"but only {num_gpus} GPU(s) are available on this instance.\n"
                f"Please reduce 'number_of_gpu_to_use_eagle3_training' to {num_gpus} or less, "
                f"or use a compute instance with more GPUs."
            )
            print(error_msg)
            raise ValueError(error_msg)

        torch.cuda.set_device(local_rank)
        print(f"Set CUDA device to {local_rank}")

        init_distributed(timeout=args.dist_timeout, tp_size=args.tp_size)
        print_with_rank("Initialized distributed environment")
    except Exception as e:
        print(f"ERROR: Failed to initialize distributed environment: {e}")
        import traceback
        print(f"Traceback:\n{traceback.format_exc()}")
        raise

    # Verify distributed is initialized
    if not dist.is_initialized():
        raise RuntimeError("Distributed training not properly initialized")

    # Print all input arguments for reference (after distributed init)
    print_on_rank0("=" * 80)
    print_on_rank0("Training Configuration - Input Arguments:")
    print_on_rank0("=" * 80)
    for arg, value in sorted(vars(args).items()):
        print_on_rank0(f"{arg:30s} = {value}")
    print_on_rank0("=" * 80)
    args.dp_size = dist.get_world_size() // args.tp_size
    args.draft_accumulation_steps = (
        args.draft_global_batch_size // args.dp_size // args.draft_micro_batch_size
    )
    assert (
        args.draft_accumulation_steps * args.draft_micro_batch_size * args.dp_size
        == args.draft_global_batch_size
    ), f"draft_global_batch_size={args.draft_global_batch_size} must be divisible by dp_size={args.dp_size} and micro_batch_size={args.draft_micro_batch_size}"
    print_with_rank(
        f"draft_accumulation_steps={args.draft_global_batch_size} // {args.dp_size} // {args.draft_micro_batch_size}={args.draft_accumulation_steps}"
    )

    # Initialize tracker based on report_to argument
    logger.info(f"Reporting metrics to {args.report_to}")

    # Handle tracker initialization
    if args.report_to == "azure_ml":
        tracker = AzureMLTrackerAdapter(args, args.output_dir)
        logger.info("AzureML tracker initialized")
    else:
        # For other tracker types, use the get_tracker_class function from specforge
        tracker_class = get_tracker_class(args.report_to)
        tracker_class.validate_args(parser, args)
        tracker = tracker_class(args, args.output_dir)
        logger.info(f"{args.report_to} tracker initialized")

    # Find the correct model path containing config.json
    args.target_model_path = find_model_config_path(args.target_model_path)
    print_on_rank0(f"Using target model path: {args.target_model_path}")
    logger.info(f"Target model path: {args.target_model_path}")

    # Handle draft model config
    if args.draft_model_config is None:
        # Auto-generate and save config file
        logger.info("Auto-generating draft model config from target model")
        auto_config_path = create_draft_config_from_target(
            target_model_path=args.target_model_path, cache_dir=cache_dir
        )
        draft_model_config = AutoDraftModelConfig.from_file(auto_config_path)
        logger.info(f"Draft model config auto-generated at: {auto_config_path}")
    else:
        # Use provided config file
        logger.info(f"Using provided draft model config: {args.draft_model_config}")
        draft_model_config = AutoDraftModelConfig.from_file(args.draft_model_config)

    # detecting last ckpt for draft model
    draft_model_last_checkpoint = None

    if args.resume:
        # When resume is true, check for checkpoints in output folder first
        if os.path.isdir(args.output_dir):
            print_on_rank0(f"Checking for checkpoints in output directory: {args.output_dir}")
            draft_model_last_checkpoint = get_last_checkpoint(args.output_dir)

            if draft_model_last_checkpoint:
                print_on_rank0(f"Found checkpoint in output directory: {draft_model_last_checkpoint}")
                logger.info(f"Resuming training from checkpoint: {draft_model_last_checkpoint}")
            elif args.resume_from_checkpoint:
                # No checkpoint in output folder, use user input checkpoint
                if os.path.isdir(args.resume_from_checkpoint) and os.path.exists(os.path.join(args.resume_from_checkpoint, "config.json")):
                    draft_model_last_checkpoint = args.resume_from_checkpoint
                    print_on_rank0(f"No checkpoint in output directory. Using user-provided checkpoint: {draft_model_last_checkpoint}")
                    logger.info(f"Using user-provided checkpoint: {draft_model_last_checkpoint}")
                else:
                    error_msg = f"ERROR: resume=True but user-provided checkpoint path is invalid: {args.resume_from_checkpoint}"
                    print_on_rank0(error_msg)
                    logger.error(error_msg)
                    raise ValueError(error_msg)
            else:
                # No checkpoint in output folder and no user input
                error_msg = f"ERROR: resume=True but no checkpoint found in output directory ({args.output_dir}) and no resume_from_checkpoint provided"
                print_on_rank0(error_msg)
                logger.error(error_msg)
                raise ValueError(error_msg)
        elif args.resume_from_checkpoint:
            # Output directory doesn't exist, use user input checkpoint
            if os.path.isdir(args.resume_from_checkpoint) and os.path.exists(os.path.join(args.resume_from_checkpoint, "config.json")):
                draft_model_last_checkpoint = args.resume_from_checkpoint
                print_on_rank0(f"Output directory does not exist. Using user-provided checkpoint: {draft_model_last_checkpoint}")
                logger.info(f"Using user-provided checkpoint: {draft_model_last_checkpoint}")
            else:
                error_msg = f"ERROR: resume=True but user-provided checkpoint path is invalid: {args.resume_from_checkpoint}"
                print_on_rank0(error_msg)
                logger.error(error_msg)
                raise ValueError(error_msg)
        else:
            # No output directory and no user input
            error_msg = f"ERROR: resume=True but output directory does not exist ({args.output_dir}) and no resume_from_checkpoint provided"
            print_on_rank0(error_msg)
            logger.error(error_msg)
            raise ValueError(error_msg)
    else:
        # When resume is false, check if user wants to initialize from a checkpoint
        if args.resume_from_checkpoint:
            if os.path.isdir(args.resume_from_checkpoint) and os.path.exists(os.path.join(args.resume_from_checkpoint, "config.json")):
                draft_model_last_checkpoint = args.resume_from_checkpoint
                print_on_rank0(f"Initializing draft model from user-provided checkpoint: {draft_model_last_checkpoint}")
                logger.info(f"Initializing draft model from checkpoint: {draft_model_last_checkpoint}")
            else:
                error_msg = f"ERROR: Invalid resume_from_checkpoint path provided: {args.resume_from_checkpoint}"
                print_on_rank0(error_msg)
                logger.error(error_msg)
                raise ValueError(error_msg)

    # build target and draft model
    if args.tp_size > 1:
        # check if the target model has tp_plan
        config = AutoConfig.from_pretrained(args.target_model_path)

        if type(config) in AutoDistributedTargetModel._model_mapping:
            target_model = AutoDistributedTargetModel.from_pretrained(
                pretrained_model_name_or_path=args.target_model_path,
                torch_dtype=torch.bfloat16,
                device="cuda",
                local_files_only=True,
            ).eval()
        else:
            target_model = AutoModelForCausalLM.from_pretrained(
                args.target_model_path,
                tp_plan="auto",
                tp_size=args.tp_size,
                torch_dtype=torch.bfloat16,
                device_mesh=get_tp_device_mesh(),
            ).eval()
    else:
        if args.is_vlm and getattr(draft_model_config, "target_model_type", None) == "qwen2_5_vl":
            from transformers import Qwen2_5_VLForConditionalGeneration

            target_model = (
                Qwen2_5_VLForConditionalGeneration.from_pretrained(
                    pretrained_model_name_or_path=args.target_model_path,
                    torch_dtype=torch.bfloat16,
                )
                .eval()
                .cuda()
            )
        else:
            target_model = (
                AutoModelForCausalLM.from_pretrained(
                    pretrained_model_name_or_path=args.target_model_path,
                    torch_dtype=torch.bfloat16,
                    cache_dir=cache_dir,
                )
                .eval()
                .cuda()
            )

    for p in target_model.parameters():
        p.requires_grad = False

    print_with_rank("Initialized target model")

    # load model with resume
    if draft_model_last_checkpoint:
        draft_model = (
            AutoEagle3DraftModel.from_pretrained(
                draft_model_last_checkpoint, attention_backend=args.attention_backend,
                torch_dtype=torch.bfloat16
            )
            .cuda()

        )
    else:
        draft_model = (
            AutoEagle3DraftModel.from_config(
                draft_model_config, attention_backend=args.attention_backend,
                torch_dtype=torch.bfloat16
            )
            .cuda()
        )
    draft_model.load_embedding(args.target_model_path, embedding_key=args.embedding_key)
    draft_model.freeze_embedding()
    print_with_rank("Initialized draft model")

    # build dataloaders
    tokenizer = AutoTokenizer.from_pretrained(args.target_model_path)
    processor = None  # VLM not supported in this version

    # convert to dataloader
    cache_params_string = (
        f"{args.dataset_train_split}-"
        f"{args.max_length}-"
        f"{args.chat_template}-"
        f"{args.target_model_path}"  # Tokenizer may also different
    )
    cache_key = hashlib.md5(cache_params_string.encode()).hexdigest()

    print_on_rank0(f"Loading training dataset from: {args.dataset_train_split}")
    try:
        dataset_dict = load_dataset("json", data_files=args.dataset_train_split)
        print_on_rank0(f"Available splits: {list(dataset_dict.keys())}")
        train_dataset = dataset_dict["train"]
        print_on_rank0(f"Loaded training dataset with {len(train_dataset)} examples")
        if len(train_dataset) > 0:
            print_on_rank0(f"Sample dataset keys: {list(train_dataset[0].keys())}")
            print_on_rank0(f"First example preview: {str(train_dataset[0])[:200]}...")
    except Exception as e:
        print_on_rank0(f"ERROR: Failed to load training dataset: {e}")
        raise

    with rank_0_priority():
        print_on_rank0("Building Eagle3 dataset...")
        print_on_rank0(f"Dataset build parameters:")
        print_on_rank0(f"  chat_template: {args.chat_template}")
        print_on_rank0(f"  max_length: {args.max_length}")
        print_on_rank0(f"  is_vlm: {args.is_vlm}")
        print_on_rank0(f"  is_preformatted: {args.is_preformatted}")
        print_on_rank0(f"  num_proc: {args.build_dataset_num_proc}")
        print_on_rank0(f"  cache_key: {cache_key}")

        try:
            train_eagle3_dataset = build_eagle3_dataset(
                dataset=train_dataset,
                tokenizer=tokenizer,
                chat_template=args.chat_template,
                max_length=args.max_length,
                cache_dir=os.path.join(cache_dir, "processed_dataset"),
                cache_key=cache_key,
                is_vlm=args.is_vlm,
                is_preformatted=args.is_preformatted,
                processor=processor,
                num_proc=args.build_dataset_num_proc,
            )
            print_on_rank0(f"Successfully built Eagle3 dataset with {len(train_eagle3_dataset)} examples")
        except Exception as e:
            print_on_rank0(f"ERROR: Failed to build Eagle3 dataset: {e}")
            print_on_rank0(f"Exception type: {type(e).__name__}")
            import traceback
            print_on_rank0(f"Full traceback:\n{traceback.format_exc()}")
            raise
        vocab_mapping_path = generate_vocab_mapping_file(
            dataset=train_eagle3_dataset,
            target_vocab_size=draft_model_config.vocab_size,
            draft_vocab_size=draft_model_config.draft_vocab_size,
            cache_dir=os.path.join(cache_dir, "vocab_mapping"),
            cache_key=cache_key,
        )
    train_dataloader = prepare_dp_dataloaders(
        train_eagle3_dataset,
        args.draft_micro_batch_size,
        num_workers=4,
        shuffle=True,
        process_group=get_dp_group(),
        is_vlm=args.is_vlm,
    )
    print_with_rank("Initialized train dataloader")

    # Calculate total steps if not provided
    if args.total_steps is None:
        steps_per_epoch = math.ceil(
            len(train_dataloader) / args.draft_accumulation_steps
        )
        args.total_steps = args.num_epochs * steps_per_epoch
        print_with_rank(
            f"Auto-calculated total_steps: {args.total_steps} (num_epochs={args.num_epochs} * steps_per_epoch={steps_per_epoch})"
        )
        logger.info(f"Auto-calculated total_steps: {args.total_steps}")
    else:
        print_with_rank(f"Using provided total_steps: {args.total_steps}")
        logger.info(f"Using provided total_steps: {args.total_steps}")

    # we load the vocab mapping then
    draft_model.load_vocab_mapping(vocab_mapping_path)
    print_with_rank("Loaded vocab mapping")

    if args.dataset_validation_split is not None:
        eval_dataset = load_dataset("json", data_files=args.dataset_validation_split)["train"]
        eval_eagle3_dataset = build_eagle3_dataset(
            eval_dataset,
            tokenizer,
            args.chat_template,
            args.max_length,
            is_vlm=args.is_vlm,
            processor=processor,
            num_proc=args.build_dataset_num_proc,
            is_preformatted=args.is_preformatted,
        )
        eval_dataloader = prepare_dp_dataloaders(
            eval_eagle3_dataset,
            args.batch_size,
            num_workers=4,
            shuffle=False,
            process_group=get_dp_group(),
            is_vlm=args.is_vlm,
        )
        print_with_rank("Initialized eval dataloader")

    # build Eagle3 model
    # broadcast draft model
    if args.is_vlm and getattr(draft_model_config, "target_model_type", None) == "qwen2_5_vl":
        eagle3_model = QwenVLOnlineEagle3Model(
            target_model=target_model,
            draft_model=draft_model,
            processor=processor,
            length=args.ttt_length,
            attention_backend=args.attention_backend,
        )
    else:
        eagle3_model = OnlineEagle3Model(
            target_model=target_model,
            draft_model=draft_model,
            length=args.ttt_length,
            attention_backend=args.attention_backend,
        )
    # eagle3_model = DDP(eagle3_model, find_unused_parameters=True)
    eagle3_model = FSDP(
        eagle3_model,
        use_orig_params=True,
        mixed_precision=MixedPrecision(
            param_dtype=torch.bfloat16,
            buffer_dtype=torch.bfloat16,
        ),
        sharding_strategy=ShardingStrategy.SHARD_GRAD_OP,
        ignored_modules=[target_model],
        process_group=get_dp_group(),
    )
    print_with_rank("Initialized Eagle3 FSDP model")

    # build other components
    optimizer = BF16Optimizer(
        draft_model,
        lr=args.learning_rate,
        max_grad_norm=args.max_grad_norm,
        warmup_ratio=args.warmup_ratio,
        total_steps=args.total_steps,
    )
    print_with_rank("Initialized optimizer and scheduler")

    # global_step
    global_step = 0
    start_epoch = 0
    # Only load training state (optimizer, epoch) when resume=True
    # When resume=False but resume_from_checkpoint is provided, we only load model weights (done above)
    if args.resume and draft_model_last_checkpoint is not None:
        print_on_rank0(
            f"Resuming draft model training from checkpoint: {draft_model_last_checkpoint}"
        )
        state_path = os.path.join(draft_model_last_checkpoint, "training_state.pt")

        if os.path.exists(state_path):
            state = torch.load(state_path, map_location="cpu", weights_only=False)
            optimizer.load_state_dict(state)
            start_epoch = state["epoch"] + 1
            global_step = state.get("global_step", 0)
            print_on_rank0(f"Resuming from epoch {start_epoch}")
            logger.info(f"Loaded training state: epoch={start_epoch}, global_step={global_step}")
        else:
            print_on_rank0(
                f"Warning: Checkpoint directory {draft_model_last_checkpoint} found, but training_state.pt is missing. Starting from scratch."
            )
            logger.warning(f"training_state.pt not found in {draft_model_last_checkpoint}, starting from epoch 0")
    elif draft_model_last_checkpoint is not None:
        print_on_rank0(f"Draft model initialized from checkpoint {draft_model_last_checkpoint}, but starting training from epoch 0 (resume=False)")
        logger.info(f"Draft model initialized from checkpoint, starting fresh training from epoch 0")

    dist.barrier()

    last_time = time.time()

    # start running
    print_on_rank0(f"Starting training from epoch {start_epoch}")
    logger.info(f"Starting training from epoch {start_epoch} to {args.num_epochs}")
    logger.info(f"Training configuration: num_epochs={args.num_epochs}, batch_size={args.draft_micro_batch_size}, "
                f"learning_rate={args.learning_rate}, total_steps={args.total_steps}")
    batch_index, log_dict = 0, defaultdict(float)

    for epoch in range(start_epoch, args.num_epochs):
        # Run training
        train_dataloader.sampler.set_epoch(epoch + 1)
        draft_model.train()
        epoch_acces = [[] for _ in range(eagle3_model.module.length)]
        epoch_plosses = [[] for _ in range(eagle3_model.module.length)]

        if dist.get_rank() == 0:
            progress_bar = tqdm(
                train_dataloader, desc=f"Training Epoch {epoch}", leave=True
            )
        else:
            progress_bar = train_dataloader

        for data in progress_bar:
            batch_index += 1
            if args.profile:
                if batch_index == args.profile_start_step:
                    print("Start profile")
                    torch_profiler = torch.profiler.profile(
                        activities=[
                            torch.profiler.ProfilerActivity.CPU,
                            torch.profiler.ProfilerActivity.CUDA,
                        ],
                        with_stack=True,
                        record_shapes=args.profile_record_shapes,
                    )
                    torch_profiler.start()
                if batch_index == args.profile_start_step + args.profile_num_steps:
                    profiler_dir = os.environ.get("SGLANG_TORCH_PROFILER_DIR", "./profiler_output")
                    os.makedirs(profiler_dir, exist_ok=True)
                    output_path = os.path.join(
                        profiler_dir,
                        f"debug_rank{torch.distributed.get_rank()}_{time.time()}.trace.json.gz",
                    )
                    print(f"End profile {output_path=}")
                    torch_profiler.stop()
                    torch_profiler.export_chrome_trace(output_path)

            if args.is_vlm:
                plosses, _, acces = eagle3_model(
                    input_ids=data["input_ids"].cuda(),
                    attention_mask=data["attention_mask"].cuda(),
                    loss_mask=data["loss_mask"].cuda(),
                    pixel_values=data["pixel_values"].cuda(),
                    image_grid_thw=data["image_grid_thw"].cuda(),
                )
            else:
                plosses, _, acces = eagle3_model(
                    input_ids=data["input_ids"].cuda(),
                    attention_mask=data["attention_mask"].cuda(),
                    loss_mask=data["loss_mask"].cuda(),
                )
            acces = torch.stack(acces).cpu().tolist()

            # calculate weighted loss
            ploss_weight = [0.8**i for i in range(len(plosses))]
            ploss = (
                sum([ploss_weight[i] * plosses[i] for i in range(len(plosses))])
                / args.draft_accumulation_steps
            )
            ploss.backward()
            log_dict["train/lr"] = optimizer.get_learning_rate()
            for i in range(len(plosses)):
                log_dict[f"train/ploss_{i}"] += (
                    plosses[i].item() / args.draft_accumulation_steps
                )
            for i in range(len(acces)):
                log_dict[f"train/acc_{i}"] += acces[i] / args.draft_accumulation_steps
            if batch_index % args.draft_accumulation_steps == 0:
                optimizer.step()
                global_step += 1
                if global_step % args.log_steps == 0:
                    tracker.log(log_dict, step=global_step)
                log_dict = defaultdict(float)

            epoch_acces = [epoch_acces[i] + [acces[i]] for i in range(len(acces))]
            epoch_plosses = [
                epoch_plosses[i] + [plosses[i].item()] for i in range(len(plosses))
            ]

            if args.verbose:
                print(
                    f"[{dist.get_rank()}] time={(time.time() - last_time):.3}s shape={data['input_ids'].shape}"
                )
                last_time = time.time()

            if dist.get_rank() == 0:
                avg_loss = sum(pl.item() for pl in plosses) / len(plosses)
                avg_acc = sum(acces) / len(acces)
                progress_bar.set_postfix(
                    {"loss": f"{avg_loss:.2f}", "acc": f"{avg_acc:.2f}"}
                )

        epoch_logdict = {}
        for i in range(len(epoch_acces)):
            acc_i = torch.tensor(epoch_acces[i]).cuda().mean()
            dist.all_reduce(acc_i)
            acc_i = (acc_i / dist.get_world_size()).item()
            epoch_logdict[f"train/epoch_acc_{i}"] = acc_i
            print_on_rank0(
                f"Train Epoch [{epoch + 1}/{args.num_epochs}], position {i},  Acc: {acc_i:.2f}"
            )

        for i in range(len(epoch_plosses)):
            loss_i = torch.tensor(epoch_plosses[i]).cuda().mean()
            dist.all_reduce(loss_i)
            loss_i = (loss_i / dist.get_world_size()).item()
            epoch_logdict[f"train/epoch_ploss_{i}"] = loss_i
            print_on_rank0(
                f"Train Epoch [{epoch + 1}/{args.num_epochs}], position {i}, pLoss: {loss_i:.2f}"
            )
        tracker.log(epoch_logdict, step=global_step)

        # run evaluation
        if args.dataset_validation_split is not None and epoch % args.eval_interval == 0:
            # Run evaluation
            draft_model.eval()
            eval_acces = [[] for _ in range(eagle3_model.module.length)]
            eval_plosses = [[] for _ in range(eagle3_model.module.length)]

            for data in tqdm(eval_dataloader, desc=f"Evaluating Epoch {epoch}"):
                if args.is_vlm:
                    with torch.no_grad():
                        plosses, _, acces = eagle3_model(
                            input_ids=data["input_ids"].cuda(),
                            attention_mask=data["attention_mask"].cuda(),
                            loss_mask=data["loss_mask"].cuda(),
                            pixel_values=data["pixel_values"].cuda(),
                            image_grid_thw=data["image_grid_thw"].cuda(),
                        )
                else:
                    with torch.no_grad():
                        plosses, _, acces = eagle3_model(
                            input_ids=data["input_ids"].cuda(),
                            attention_mask=data["attention_mask"].cuda(),
                            loss_mask=data["loss_mask"].cuda(),
                        )
                acces = torch.stack(acces).cpu().tolist()

                eval_acces = [eval_acces[i] + [acces[i]] for i in range(len(acces))]
                eval_plosses = [
                    eval_plosses[i] + [plosses[i].item()] for i in range(len(plosses))
                ]

            # Log epoch-level evaluation metrics
            eval_logdict = {}
            for i in range(len(eval_acces)):
                acc_i = torch.tensor(eval_acces[i]).cuda().mean()
                dist.all_reduce(acc_i)
                acc_i = (acc_i / dist.get_world_size()).item()
                eval_logdict[f"eval/epoch_acc_{i}"] = acc_i
                print_on_rank0(
                    f"Eval Epoch [{epoch + 1}/{args.num_epochs}], position {i},  Acc: {acc_i:.2f}"
                )

            for i in range(len(eval_plosses)):
                loss_i = torch.tensor(eval_plosses[i]).cuda().mean()
                dist.all_reduce(loss_i)
                loss_i = (loss_i / dist.get_world_size()).item()
                eval_logdict[f"eval/epoch_ploss_{i}"] = loss_i
                print_on_rank0(
                    f"Eval Epoch [{epoch + 1}/{args.num_epochs}], position {i}, pLoss: {loss_i:.2f}"
                )
            tracker.log(eval_logdict, step=global_step)

        if epoch % args.save_interval == 0:
            # Save the model
            epoch_output_dir = os.path.join(args.output_dir, f"epoch_{epoch}")
            logger.info(f"Saving checkpoint for epoch {epoch} to {epoch_output_dir}")

            if dist.get_rank() == 0:
                os.makedirs(epoch_output_dir, exist_ok=True)
            dist.barrier()

            with FSDP.state_dict_type(eagle3_model, StateDictType.FULL_STATE_DICT):
                model_state_dict = eagle3_model.state_dict()
                state_to_save = {
                    "epoch": epoch,
                    "global_step": global_step,
                    "args": args,
                }
                state_to_save.update(optimizer.state_dict())
                draft_model_state_dict = {
                    k.replace("draft_model.", ""): v
                    for k, v in model_state_dict.items()
                    if "draft_model." in k and "embed" not in k.lower()
                }

                if dist.get_rank() == 0:
                    torch.save(
                        state_to_save,
                        os.path.join(epoch_output_dir, "training_state.pt"),
                    )
                    print_on_rank0(
                        f"Saved full training state to {epoch_output_dir}/training_state.pt"
                    )
                    draft_model.save_pretrained(
                        epoch_output_dir,
                        state_dict=draft_model_state_dict,
                    )
                    print_on_rank0(f"Saved model configuration to {epoch_output_dir}")
                dist.barrier()

    # Close the tracker
    tracker.close()
    logger.info("Training completed successfully")
    destroy_distributed()


if __name__ == "__main__":
    main()
