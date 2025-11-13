#!/usr/bin/env python
"""
Sweep Checkpoint Evaluations Component

This component:
1. Lists the directory structure of checkpoints
2. Sweeps through multiple checkpoints based on explore_pattern
3. Evaluates each checkpoint using vLLM with step-based logging
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import List, Tuple
from datetime import datetime

# Import functions from other modules instead of subprocess calls
from list_checkpoint_dirs import list_directory_contents, format_size
from preprocess_checkpoint import preprocess_checkpoint

# Import vLLM evaluation - will be imported at runtime to avoid import errors
# from vllm_evaluation_step_logging import run_evaluation


def str2bool(v):
    """Convert string to boolean for argparse compatibility."""
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def log_section_header(title: str):
    """Log section header with timestamp."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"\n{'='*80}")
    print(f"[{timestamp}] {title}")
    print(f"{'='*80}\n")


def get_azureml_run():
    """Get AzureML Run context if available."""
    try:
        from azureml.core.run import Run
        azureml_run = Run.get_context()
        if azureml_run and "OfflineRun" not in azureml_run.id:
            return azureml_run
    except ImportError:
        print("Warning: azureml-core not available - AzureML logging disabled")
    except Exception as e:
        print(f"Warning: Failed to get AzureML run context: {e}")
    return None


def list_checkpoint_directory_structure(base_path: str, max_depth: int = 3):
    """List and print the checkpoint directory structure."""
    log_section_header("LISTING CHECKPOINT DIRECTORY STRUCTURE")
    
    print(f"Base Path: {base_path}")
    print(f"Max Depth: {max_depth}")
    print(f"{'='*80}\n")
    
    try:
        if not os.path.exists(base_path):
            print(f"❌ ERROR: Path does not exist: {base_path}")
            return
        
        print(f"Path exists: {base_path}")
        print(f"Is Directory: {os.path.isdir(base_path)}")
        
        if os.path.isdir(base_path):
            print(f"\n📁 Directory Contents:\n")
            list_directory_contents(base_path, max_depth=max_depth)
            
            # Count files and directories
            total_files = 0
            total_dirs = 0
            total_size = 0
            
            for root, dirs, files in os.walk(base_path):
                total_dirs += len(dirs)
                total_files += len(files)
                for file in files:
                    try:
                        total_size += os.path.getsize(os.path.join(root, file))
                    except:
                        pass
            
            print(f"\n{'='*80}")
            print(f"📊 Summary:")
            print(f"  Total Directories: {total_dirs}")
            print(f"  Total Files: {total_files}")
            print(f"  Total Size: {format_size(total_size)}")
            print(f"{'='*80}\n")
        else:
            size = os.path.getsize(base_path)
            print(f"\n📄 File Size: {format_size(size)}")
            print(f"{'='*80}\n")
            
    except Exception as e:
        print(f"❌ Error listing directory: {e}")
        import traceback
        traceback.print_exc()


def resolve_checkpoint_path(base_path: str, pattern: str, checkpoint_value: str) -> str:
    """Resolve the full checkpoint path from pattern and value."""
    # Replace {checkpoint} placeholder with actual value
    relative_path = pattern.replace("{checkpoint}", str(checkpoint_value))
    full_path = os.path.join(base_path, relative_path)
    return full_path


def evaluate_checkpoint(
    checkpoint_path: str,
    checkpoint_value: str,
    validation_file: str,
    output_dir: str,
    intermediate_dir: str,
    use_lora_adapters: bool,
    base_model_path: str,
    hf_model_id: str,
    args: argparse.Namespace,
    azureml_run,
    checkpoint_source: str = "base_path_1"
):
    """Evaluate a single checkpoint using vLLM."""
    print("\n" + "=" * 80)
    print(f"EVALUATING CHECKPOINT: {checkpoint_value}")
    print(f"Source: {checkpoint_source}")
    print("=" * 80)
    print(f"Mode: {'LoRA Adapter' if use_lora_adapters else 'Full Checkpoint'}")
    if use_lora_adapters:
        if hf_model_id:
            print(f"Base Model (HF): {hf_model_id}")
        elif base_model_path:
            print(f"Base Model (Local): {base_model_path}")
        print(f"LoRA Adapter: {checkpoint_path}")
    else:
        print(f"Checkpoint path: {checkpoint_path if checkpoint_path else 'Using HF Model ID'}")
    print(f"Output directory: {output_dir}")
    print(f"Intermediate directory: {intermediate_dir}")
    print("=" * 80 + "\n")
    
    # Validate paths/IDs based on mode
    if use_lora_adapters:
        # LoRA mode: checkpoint_path must exist, and base model is required
        if not os.path.exists(checkpoint_path):
            print(f"❌ ERROR: Adapter path does not exist: {checkpoint_path}")
            print(f"Skipping checkpoint {checkpoint_value}")
            return False
        # LoRA mode validation
        # Validate that either base_model_path or hf_model_id is provided
        if not base_model_path and not hf_model_id:
            print(f"❌ ERROR: Either base_model_path or hf_model_id is required when use_lora_adapters is true")
            return False
        
        # Validate mutual exclusivity
        if base_model_path and hf_model_id:
            print(f"⚠️  WARNING: Both base_model_path and hf_model_id provided. Using base_model_path and ignoring hf_model_id")
            actual_base_model = base_model_path
        elif hf_model_id:
            print(f"✅ Using Hugging Face model: {hf_model_id}")
            actual_base_model = hf_model_id
        else:
            # Validate local path exists
            if not os.path.exists(base_model_path):
                print(f"❌ ERROR: Base model path does not exist: {base_model_path}")
                return False
            print(f"✅ Base model found: {base_model_path}")
            actual_base_model = base_model_path
        
        print(f"✅ LoRA adapter found: {checkpoint_path}")
        
        # For LoRA: use adapter path and base model
        eval_checkpoint_path = checkpoint_path
        eval_base_model = actual_base_model
    else:
        # Full model mode: use checkpoint_path if exists, otherwise use HF model
        if checkpoint_path and os.path.exists(checkpoint_path):
            # Local checkpoint exists - preprocess it
            log_section_header(f"PREPROCESSING CHECKPOINT {checkpoint_value}")
            
            intermediate_checkpoint_path = os.path.join(intermediate_dir, f"checkpoint_{checkpoint_value}")
            
            print(f"Original Path: {checkpoint_path}")
            print(f"Intermediate Path: {intermediate_checkpoint_path}")
            print(f"{'='*80}\n")
            
            # Path to use for evaluation (either intermediate or original)
            eval_checkpoint_path = checkpoint_path
            eval_base_model = None
            
            try:
                # Call preprocessing function directly
                success, path_to_use = preprocess_checkpoint(checkpoint_path, intermediate_checkpoint_path)
                
                if success:
                    eval_checkpoint_path = path_to_use
                    if path_to_use != checkpoint_path:
                        print(f"\n✅ Using preprocessed checkpoint from: {eval_checkpoint_path}")
                    else:
                        print(f"\nℹ️  No preprocessing needed, using original: {checkpoint_path}")
                else:
                    print(f"\n⚠️  Warning: Preprocessing failed, using original checkpoint path")
                    eval_checkpoint_path = checkpoint_path
                    
            except Exception as e:
                print(f"\n❌ ERROR during preprocessing: {e}")
                import traceback
                traceback.print_exc()
                print(f"\n⚠️  Warning: Preprocessing failed, using original checkpoint path")
                eval_checkpoint_path = checkpoint_path
        elif hf_model_id:
            # No local checkpoint - use HF model directly
            print(f"✅ Using Hugging Face model directly: {hf_model_id}")
            eval_checkpoint_path = hf_model_id
            eval_base_model = None
        else:
            print(f"❌ ERROR: Checkpoint path does not exist: {checkpoint_path}")
            return False
    
    # Create checkpoint-specific output directory with source identifier
    checkpoint_output_dir = os.path.join(output_dir, f"{checkpoint_source}_checkpoint_{checkpoint_value}")
    os.makedirs(checkpoint_output_dir, exist_ok=True)
    
    # Run vLLM evaluation directly
    log_section_header(f"EVALUATING CHECKPOINT {checkpoint_value}")
    print(f"📊 Evaluation Details:")
    if use_lora_adapters:
        print(f"   Mode: LoRA Adapter")
        print(f"   Base Model: {eval_base_model}")
        print(f"   LoRA Adapter: {eval_checkpoint_path}")
    else:
        print(f"   Mode: Full Checkpoint")
        print(f"   Original Path: {checkpoint_path}")
        print(f"   Eval Path: {eval_checkpoint_path}")
    print(f"   Output Dir: {checkpoint_output_dir}")
    print(f"   Trials: {args.number_of_trials}")
    print(f"   Batch Size: {args.batch_size}")
    print(f"   Temperature: {args.temperature}")
    print(f"{'='*80}\n")
    
    try:
        # Import and run vLLM evaluation directly
        sys.path.insert(0, os.path.dirname(__file__))
        from vllm_evaluation_step_logging import main as vllm_main
        
        # Save original sys.argv
        original_argv = sys.argv
        
        # Create arguments for vLLM evaluation
        sys_argv_list = [
            'vllm_evaluation_step_logging.py',
            '--model_path', eval_checkpoint_path,
            '--validation_file', validation_file,
            '--max_prompt_length', str(args.max_prompt_length),
            '--max_response_length', str(args.max_response_length),
            '--batch_size', str(args.batch_size),
            '--temperature', str(args.temperature),
            '--top_p', str(args.top_p),
            '--tensor_parallel_size', str(args.tensor_parallel_size),
            '--gpu_memory_utilization', str(args.gpu_memory_utilization),
            '--dtype', args.dtype,
            '--extraction_method', args.extraction_method,
            '--n_gpus_per_node', str(args.n_gpus_per_node),
            '--number_of_trials', str(args.number_of_trials),
            '--output_dir', checkpoint_output_dir
        ]
        
        # Add LoRA-specific arguments if using adapters
        if use_lora_adapters:
            sys_argv_list.extend([
                '--enable_lora',
                '--lora_modules', f'adapter={eval_checkpoint_path}',
                '--max_lora_rank', str(args.max_lora_rank)
            ])
            # Override model_path to be the base model
            sys_argv_list[sys_argv_list.index('--model_path') + 1] = eval_base_model
        
        sys.argv = sys_argv_list
        
        # Call vLLM evaluation main function
        vllm_main()
        
        # Restore original sys.argv
        sys.argv = original_argv
        
        print(f"\n{'='*80}")
        print(f"✅ Successfully evaluated checkpoint {checkpoint_value}")
        print(f"{'='*80}\n")
        
        # Load and log checkpoint-level aggregate metrics to AzureML
        if azureml_run:
            try:
                aggregate_metrics_path = os.path.join(checkpoint_output_dir, "aggregate_metrics.json")
                if os.path.exists(aggregate_metrics_path):
                    with open(aggregate_metrics_path, 'r') as f:
                        agg_metrics = json.load(f)
                    
                    # Log aggregate metrics with checkpoint identifier and source
                    azureml_run.log(f"{checkpoint_source}/checkpoint_{checkpoint_value}/accuracy_mean", agg_metrics['accuracy']['mean'])
                    azureml_run.log(f"{checkpoint_source}/checkpoint_{checkpoint_value}/format_rate_mean", agg_metrics['format_rate']['mean'])
                    azureml_run.log(f"{checkpoint_source}/checkpoint_{checkpoint_value}/correct_answers_mean", agg_metrics['correct_answers']['mean'])
                    
                    if agg_metrics['number_of_trials'] > 1:
                        azureml_run.log(f"{checkpoint_source}/checkpoint_{checkpoint_value}/accuracy_std", agg_metrics['accuracy']['std'])
                    
                    print(f"Logged checkpoint {checkpoint_value} (source: {checkpoint_source}) aggregate metrics to AzureML")
            except Exception as e:
                print(f"Warning: Failed to log checkpoint metrics to AzureML: {e}")
        
        return True
        
    except Exception as e:
        print(f"\n{'='*80}")
        print(f"❌ ERROR: Failed to evaluate checkpoint {checkpoint_value}")
        print(f"   Error: {e}")
        print(f"{'='*80}\n")
        import traceback
        traceback.print_exc()
        return False


def main():
    parser = argparse.ArgumentParser(description="Sweep Checkpoint Evaluations Component")
    
    # Checkpoint sweep parameters
    parser.add_argument("--checkpoint_base_path", type=str, default=None,
                        help="Base path containing all checkpoints or LoRA adapters (optional if using hf_model_id for full models)")
    parser.add_argument("--checkpoint_base_path_2", type=str, default=None,
                        help="Second base path containing checkpoints or LoRA adapters (optional, for comparing models from different training runs)")
    parser.add_argument("--explore_pattern", type=str, 
                        default="global_step_{checkpoint}/actor/huggingface/",
                        help="Pattern to explore for checkpoint paths (only used with checkpoint_base_path)")
    parser.add_argument("--explore_pattern_2", type=str, 
                        default="global_step_{checkpoint}/actor/huggingface/",
                        help="Pattern to explore for checkpoint paths in checkpoint_base_path_2 (only used with checkpoint_base_path_2)")
    parser.add_argument("--checkpoint_values", type=str, default=None,
                        help="Comma-separated list of checkpoint values (e.g., '100,129,20'). Optional if using only hf_model_id")
    parser.add_argument("--checkpoint_values_2", type=str, default=None,
                        help="Comma-separated list of checkpoint values for checkpoint_base_path_2 (e.g., '100,129,20'). Only used with checkpoint_base_path_2")
    
    # LoRA-specific parameters
    parser.add_argument("--use_lora_adapters", type=str2bool, nargs='?', const=True, default=False,
                        help="If true, checkpoints are LoRA adapters to load with base model (accepts: True/False or flag)")
    parser.add_argument("--base_model_path", type=str, default=None,
                        help="Local base model path (mutually exclusive with hf_model_id)")
    parser.add_argument("--hf_model_id", type=str, default=None,
                        help="Hugging Face model ID (e.g., 'meta-llama/Llama-2-7b-hf', mutually exclusive with base_model_path)")
    parser.add_argument("--max_lora_rank", type=int, default=64,
                        help="Maximum LoRA rank for adapter support (default: 64)")
    
    # Evaluation parameters
    parser.add_argument("--validation_file", type=str, required=True,
                        help="Path to validation JSONL file")
    parser.add_argument("--max_prompt_length", type=int, default=2048)
    parser.add_argument("--max_response_length", type=int, default=1024)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--tensor_parallel_size", type=int, default=1)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.8)
    parser.add_argument("--dtype", type=str, default="bfloat16",
                        choices=["float16", "bfloat16", "float32"])
    parser.add_argument("--extraction_method", type=str, default="strict",
                        choices=["strict", "flexible"])
    parser.add_argument("--n_gpus_per_node", type=int, default=1)
    parser.add_argument("--number_of_trials", type=int, default=1)
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Output directory for all evaluation results")
    parser.add_argument("--intermediate_dir", type=str, required=True,
                        help="Intermediate directory for preprocessed checkpoints")
    
    args = parser.parse_args()
    
    # Validate configuration
    if args.use_lora_adapters:
        # LoRA mode: requires checkpoint_base_path and a base model
        if not args.checkpoint_base_path:
            print("❌ ERROR: --checkpoint_base_path is required when --use_lora_adapters is true")
            sys.exit(1)
        if not args.checkpoint_values:
            print("❌ ERROR: --checkpoint_values is required when --use_lora_adapters is true")
            sys.exit(1)
        if not args.base_model_path and not args.hf_model_id:
            print("❌ ERROR: Either --base_model_path or --hf_model_id is required when --use_lora_adapters is true")
            sys.exit(1)
        if args.base_model_path and args.hf_model_id:
            print("⚠️  WARNING: Both --base_model_path and --hf_model_id provided. Using --base_model_path")
    else:
        # Full model mode: need either checkpoints OR hf_model_id
        if not args.checkpoint_base_path and not args.hf_model_id:
            print("❌ ERROR: Either --checkpoint_base_path or --hf_model_id is required for full model evaluation")
            sys.exit(1)
        
        # If using checkpoints, need checkpoint_values
        if args.checkpoint_base_path and not args.checkpoint_values:
            print("❌ ERROR: --checkpoint_values is required when --checkpoint_base_path is provided")
            sys.exit(1)
        
        # If using HF model only, checkpoint_values should default to a single dummy value
        if not args.checkpoint_base_path and args.hf_model_id:
            if not args.checkpoint_values:
                args.checkpoint_values = "hf_model"
                print("ℹ️  Using HuggingFace model directly, setting checkpoint_values to 'hf_model'")
        
        if args.checkpoint_base_path and args.hf_model_id:
            print("⚠️  WARNING: Both --checkpoint_base_path and --hf_model_id provided. Using --checkpoint_base_path")
    
    # Initialize AzureML Run context
    azureml_run = get_azureml_run()
    if azureml_run:
        print("AzureML Run context found - checkpoint metrics will be logged")
    
    # Debug: Print the actual values
    print(f"\n🔍 DEBUG: use_lora_adapters = {args.use_lora_adapters} (type: {type(args.use_lora_adapters).__name__})")
    print(f"🔍 DEBUG: checkpoint_base_path = {args.checkpoint_base_path}")
    print(f"🔍 DEBUG: checkpoint_base_path_2 = {args.checkpoint_base_path_2}")
    print(f"🔍 DEBUG: hf_model_id = {args.hf_model_id}")
    print(f"🔍 DEBUG: base_model_path = {args.base_model_path}")
    
    print("\n" + "=" * 80)
    print("SWEEP CHECKPOINT EVALUATIONS")
    print("=" * 80)
    print(f"Mode: {'LoRA Adapters' if args.use_lora_adapters else 'Full Checkpoints'}")
    if args.use_lora_adapters:
        if args.hf_model_id:
            print(f"Base model (HuggingFace): {args.hf_model_id}")
        if args.base_model_path:
            print(f"Base model (Local): {args.base_model_path}")
        if args.checkpoint_base_path:
            print(f"Adapter base path 1: {args.checkpoint_base_path}")
        if args.checkpoint_base_path_2:
            print(f"Adapter base path 2: {args.checkpoint_base_path_2}")
    else:
        if args.checkpoint_base_path:
            print(f"Checkpoint base path 1: {args.checkpoint_base_path}")
        if args.checkpoint_base_path_2:
            print(f"Checkpoint base path 2: {args.checkpoint_base_path_2}")
        if args.hf_model_id:
            print(f"Using HuggingFace model: {args.hf_model_id}")
    if args.checkpoint_base_path:
        print(f"Explore pattern 1: {args.explore_pattern}")
        print(f"Checkpoint values 1: {args.checkpoint_values}")
    if args.checkpoint_base_path_2:
        print(f"Explore pattern 2: {args.explore_pattern_2}")
        print(f"Checkpoint values 2: {args.checkpoint_values_2}")
    print(f"Validation file: {args.validation_file}")
    print(f"Number of trials per checkpoint: {args.number_of_trials}")
    print(f"Output directory: {args.output_dir}")
    print(f"Intermediate directory: {args.intermediate_dir}")
    print("=" * 80)
    
    # Create output and intermediate directories
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.intermediate_dir, exist_ok=True)
    
    # Step 1: List directory structures
    if args.checkpoint_base_path:
        print("\n[STEP 1a] Listing checkpoint directory structure (base_path_1)...")
        list_checkpoint_directory_structure(args.checkpoint_base_path, max_depth=3)
    
    if args.checkpoint_base_path_2:
        print("\n[STEP 1b] Listing checkpoint directory structure (base_path_2)...")
        list_checkpoint_directory_structure(args.checkpoint_base_path_2, max_depth=3)
    
    if not args.checkpoint_base_path and not args.checkpoint_base_path_2:
        print("\n[STEP 1] Skipping directory listing (using HuggingFace model)")
    
    # Step 2: Build evaluation tasks list
    print("\n[STEP 2] Building evaluation tasks...")
    eval_tasks = []
    
    # Add tasks from checkpoint_base_path
    if args.checkpoint_values:
        checkpoint_values_1 = [val.strip() for val in args.checkpoint_values.split(',')]
        print(f"Found {len(checkpoint_values_1)} checkpoints from base_path_1: {checkpoint_values_1}")
        for checkpoint_value in checkpoint_values_1:
            eval_tasks.append({
                "checkpoint_value": checkpoint_value,
                "base_path": args.checkpoint_base_path,
                "pattern": args.explore_pattern,
                "source": "base_path_1"
            })
    
    # Add tasks from checkpoint_base_path_2
    if args.checkpoint_values_2:
        checkpoint_values_2 = [val.strip() for val in args.checkpoint_values_2.split(',')]
        print(f"Found {len(checkpoint_values_2)} checkpoints from base_path_2: {checkpoint_values_2}")
        for checkpoint_value in checkpoint_values_2:
            eval_tasks.append({
                "checkpoint_value": checkpoint_value,
                "base_path": args.checkpoint_base_path_2,
                "pattern": args.explore_pattern_2,
                "source": "base_path_2"
            })
    
    print(f"Total evaluation tasks: {len(eval_tasks)}")
    
    # Step 3: Sweep through checkpoints and evaluate
    print(f"\n[STEP 3] Sweeping through {len(eval_tasks)} checkpoint(s)...")
    
    results_summary = []
    successful_evals = 0
    failed_evals = 0
    
    for idx, task in enumerate(eval_tasks, 1):
        checkpoint_value = task["checkpoint_value"]
        base_path = task["base_path"]
        pattern = task["pattern"]
        source = task["source"]
        
        print(f"\n{'='*80}")
        print(f"Processing checkpoint {idx}/{len(eval_tasks)}: {checkpoint_value} (source: {source})")
        print(f"{'='*80}")
        
        # Resolve checkpoint path (None if using HF model directly)
        if base_path:
            checkpoint_path = resolve_checkpoint_path(
                base_path,
                pattern,
                checkpoint_value
            )
        else:
            checkpoint_path = None
        
        # Evaluate checkpoint
        success = evaluate_checkpoint(
            checkpoint_path=checkpoint_path,
            checkpoint_value=checkpoint_value,
            validation_file=args.validation_file,
            output_dir=args.output_dir,
            intermediate_dir=args.intermediate_dir,
            use_lora_adapters=args.use_lora_adapters,
            base_model_path=args.base_model_path,
            hf_model_id=args.hf_model_id,
            args=args,
            azureml_run=azureml_run,
            checkpoint_source=source
        )
        
        if success:
            successful_evals += 1
        else:
            failed_evals += 1
        
        results_summary.append({
            "checkpoint_value": checkpoint_value,
            "checkpoint_path": checkpoint_path,
            "source": source,
            "success": success,
            "output_dir": os.path.join(args.output_dir, f"{source}_checkpoint_{checkpoint_value}")
        })
    
    # Step 4: Save sweep summary
    print("\n" + "=" * 80)
    print("SWEEP SUMMARY")
    print("=" * 80)
    print(f"Total checkpoints: {len(eval_tasks)}")
    print(f"Successful evaluations: {successful_evals}")
    print(f"Failed evaluations: {failed_evals}")
    print("=" * 80)
    
    # Log summary to AzureML
    if azureml_run:
        try:
            azureml_run.log("sweep/total_checkpoints", len(eval_tasks))
            azureml_run.log("sweep/successful_evaluations", successful_evals)
            azureml_run.log("sweep/failed_evaluations", failed_evals)
            print("Logged sweep summary to AzureML")
        except Exception as e:
            print(f"Warning: Failed to log sweep summary to AzureML: {e}")
    
    # Save results summary
    summary_data = {
        "config": {
            "checkpoint_base_path": args.checkpoint_base_path,
            "checkpoint_base_path_2": args.checkpoint_base_path_2,
            "explore_pattern": args.explore_pattern,
            "explore_pattern_2": args.explore_pattern_2,
            "checkpoint_values": args.checkpoint_values,
            "checkpoint_values_2": args.checkpoint_values_2,
            "validation_file": args.validation_file,
            "number_of_trials": args.number_of_trials
        },
        "summary": {
            "total_checkpoints": len(eval_tasks),
            "successful_evaluations": successful_evals,
            "failed_evaluations": failed_evals
        },
        "results": results_summary
    }
    
    summary_path = os.path.join(args.output_dir, "sweep_summary.json")
    with open(summary_path, 'w') as f:
        json.dump(summary_data, f, indent=2)
    
    print(f"\nSweep summary saved to: {summary_path}")
    
    # Print individual checkpoint results
    print("\nCheckpoint Evaluation Results:")
    for result in results_summary:
        status = "✅ SUCCESS" if result["success"] else "❌ FAILED"
        source_label = result.get("source", "base_path_1")
        print(f"  {status} - [{source_label}] Checkpoint {result['checkpoint_value']}: {result['checkpoint_path']}")
    
    print("\n" + "=" * 80)
    print("SWEEP CHECKPOINT EVALUATIONS COMPLETED")
    print("=" * 80)
    
    # Exit with error if any evaluations failed
    if failed_evals > 0:
        print(f"\nWarning: {failed_evals} checkpoint evaluation(s) failed")
        sys.exit(0)  # Don't fail the entire job, just warn


if __name__ == "__main__":
    main()
