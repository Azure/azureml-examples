#!/usr/bin/env python
"""
Modified vLLM evaluation script that logs each trial as a step in AzureML metrics.
"""

import argparse
import json
import os
from pathlib import Path
from typing import List, Dict, Any
from tqdm import tqdm
import numpy as np

from vllm import LLM, SamplingParams
from verl.utils.reward_score.gsm8k import extract_solution
from math_verify import parse, verify


def load_validation_data(validation_file: str) -> List[Dict[str, Any]]:
    """Load validation data from JSONL file."""
    data = []
    with open(validation_file, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data


def extract_prompts(data: List[Dict[str, Any]]) -> List[str]:
    """Extract prompts from validation data."""
    prompts = []
    for item in data:
        if 'prompt' in item and isinstance(item['prompt'], list):
            prompt_parts = []
            for message in item['prompt']:
                if 'content' in message:
                    prompt_parts.append(message['content'])
            prompts.append('\n'.join(prompt_parts))
        elif 'prompt' in item and isinstance(item['prompt'], str):
            prompts.append(item['prompt'])
        elif 'problem' in item:
            prompts.append(item['problem'])
        else:
            raise ValueError(f"Unexpected prompt format in data: {item}")
    return prompts


def extract_ground_truths(data: List[Dict[str, Any]]) -> List[str]:
    """Extract ground truth answers from validation data."""
    ground_truths = []
    for item in data:
        if 'reward_model' in item and 'ground_truth' in item['reward_model']:
            ground_truths.append(item['reward_model']['ground_truth'])
        elif 'extra_info' in item and 'answer' in item['extra_info']:
            ground_truths.append(item['extra_info']['answer'])
        elif 'solution' in item:
            ground_truths.append(item['solution'])
        else:
            ground_truths.append(None)
    return ground_truths


def evaluate_responses(
    responses: List[str],
    ground_truths: List[str],
    prompts: List[str] = None,
    extraction_method: str = "strict"
) -> Dict[str, Any]:
    """Evaluate responses against ground truths."""
    total = len(responses)
    correct = 0
    format_correct = 0
    no_answer = 0
    
    results = []
    failed_predictions = []
    
    for idx, (response, ground_truth) in enumerate(zip(responses, ground_truths)):
        extracted_answer = extract_solution(response, method=extraction_method)
        
        is_correct = False
        has_format = extracted_answer is not None
        
        if extracted_answer is None:
            no_answer += 1
            status = "no_answer"
        elif ground_truth is not None:
            gold = parse(ground_truth)
            answer = parse(extracted_answer)
            if verify(gold, answer):
                correct += 1
                format_correct += 1
                is_correct = True
                status = "correct"
            else:
                format_correct += 1
                status = "wrong_answer"
        else:
            status = "no_ground_truth"
        
        result = {
            "index": idx,
            "response": response,
            "extracted_answer": extracted_answer,
            "ground_truth": ground_truth,
            "is_correct": is_correct,
            "has_format": has_format,
            "status": status
        }
        
        if prompts is not None and idx < len(prompts):
            result["prompt"] = prompts[idx]
        
        results.append(result)
        
        if not is_correct and prompts is not None and idx < len(prompts):
            failed_predictions.append({
                "index": idx,
                "prompt": prompts[idx],
                "response": response,
                "extracted_answer": extracted_answer,
                "ground_truth": ground_truth,
                "status": status
            })
    
    accuracy = correct / total if total > 0 else 0.0
    format_rate = format_correct / total if total > 0 else 0.0
    no_answer_rate = no_answer / total if total > 0 else 0.0
    
    metrics = {
        "total_samples": total,
        "correct_answers": correct,
        "format_correct": format_correct,
        "no_answer": no_answer,
        "accuracy": accuracy,
        "format_rate": format_rate,
        "no_answer_rate": no_answer_rate
    }
    
    return {"metrics": metrics, "detailed_results": results, "failed_predictions": failed_predictions}


def find_model_config(model_path: str) -> str:
    """Find the directory containing config.json in model_path or its subdirectories."""
    model_path = Path(model_path)
    
    if (model_path / "config.json").exists():
        print(f"Found config.json in: {model_path}")
        return str(model_path)
    
    print(f"Searching for config.json in subdirectories of: {model_path}")
    for config_file in model_path.rglob("config.json"):
        config_dir = config_file.parent
        print(f"Found config.json in: {config_dir}")
        return str(config_dir)
    
    print(f"Warning: config.json not found in {model_path} or its subdirectories")
    print(f"Using original model_path: {model_path}")
    return str(model_path)


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


def main():
    parser = argparse.ArgumentParser(description="vLLM Evaluation Component with Step Logging")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the model")
    parser.add_argument("--validation_file", type=str, required=True, help="Path to validation JSONL file")
    parser.add_argument("--max_prompt_length", type=int, default=2048, help="Maximum prompt length")
    parser.add_argument("--max_response_length", type=int, default=1024, help="Maximum response length")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for evaluation")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature")
    parser.add_argument("--top_p", type=float, default=0.9, help="Top-p sampling")
    parser.add_argument("--tensor_parallel_size", type=int, default=1, help="Tensor parallel size")
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.8, help="GPU memory utilization")
    parser.add_argument("--dtype", type=str, default="bfloat16", choices=["float16", "bfloat16", "float32"])
    parser.add_argument("--extraction_method", type=str, default="strict", choices=["strict", "flexible"])
    parser.add_argument("--n_gpus_per_node", type=int, default=1, help="Number of GPUs per node")
    parser.add_argument("--number_of_trials", type=int, default=1, help="Number of evaluation trials to run")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for results")
    parser.add_argument("--enable_lora", action="store_true", help="Enable LoRA adapter support")
    parser.add_argument("--lora_modules", type=str, help="LoRA modules in format: name=path,name=path,...")
    parser.add_argument("--max_lora_rank", type=int, default=64, help="Maximum LoRA rank (default: 64)")
    
    args = parser.parse_args()
    
    # Initialize AzureML Run context
    azureml_run = get_azureml_run()
    if azureml_run:
        print("AzureML Run context found - metrics will be logged as steps to AzureML")
    
    # Find the actual model directory containing config.json
    args.model_path = find_model_config(args.model_path)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("=" * 80)
    print("vLLM Evaluation Component (Step-based Logging)")
    print("=" * 80)
    print(f"Model path: {args.model_path}")
    print(f"Validation file: {args.validation_file}")
    print(f"Extraction method: {args.extraction_method}")
    print(f"Number of trials: {args.number_of_trials}")
    print(f"Output directory: {args.output_dir}")
    print("=" * 80)
    
    # Load validation data
    print("\n[1/5] Loading validation data...")
    validation_data = load_validation_data(args.validation_file)
    print(f"Loaded {len(validation_data)} samples")
    
    # Extract prompts and ground truths
    print("\n[2/5] Extracting prompts and ground truths...")
    prompts = extract_prompts(validation_data)
    ground_truths = extract_ground_truths(validation_data)
    print(f"Extracted {len(prompts)} prompts")
    
    # Initialize vLLM
    print("\n[3/5] Initializing vLLM...")
    
    # Prepare LLM kwargs
    llm_kwargs = {
        "model": args.model_path,
        "tensor_parallel_size": args.tensor_parallel_size,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "dtype": args.dtype,
        "trust_remote_code": True
    }
    
    # Parse LoRA modules if enabled (but don't add to llm_kwargs)
    lora_modules_list = []
    if args.enable_lora:
        print("LoRA adapter support enabled")
        llm_kwargs["enable_lora"] = True
        llm_kwargs["max_lora_rank"] = args.max_lora_rank
        print(f"Max LoRA rank: {args.max_lora_rank}")
        
        if args.lora_modules:
            # Parse lora_modules format: name=path,name=path,...
            for module_spec in args.lora_modules.split(','):
                if '=' in module_spec:
                    name, path = module_spec.split('=', 1)
                    lora_modules_list.append({
                        "name": name.strip(),
                        "path": path.strip()
                    })
                    print(f"  LoRA module: {name.strip()} -> {path.strip()}")
    
    llm = LLM(**llm_kwargs)
    
    # Set sampling parameters
    sampling_params = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_response_length
    )
    
    # Prepare LoRA request if enabled
    lora_request = None
    if args.enable_lora and lora_modules_list:
        from vllm.lora.request import LoRARequest
        # Use the first LoRA module (typically there's only one)
        first_module = lora_modules_list[0]
        lora_request = LoRARequest(
            lora_name=first_module["name"],
            lora_int_id=1,
            lora_path=first_module["path"]
        )
        print(f"Using LoRA adapter: {first_module['name']} from {first_module['path']}")
    
    # Run multiple trials
    print(f"\n[4/5] Running {args.number_of_trials} evaluation trial(s)...")
    all_trial_metrics = []
    all_trial_results = []
    
    for trial_idx in range(args.number_of_trials):
        print(f"\n--- Trial {trial_idx + 1}/{args.number_of_trials} ---")
        
        # Generate responses
        print(f"Generating responses with batch_size={args.batch_size}...")
        if lora_request:
            outputs = llm.generate(prompts, sampling_params, lora_request=lora_request)
        else:
            outputs = llm.generate(prompts, sampling_params)
        
        # Extract generated texts
        responses = [output.outputs[0].text for output in outputs]
        print(f"Generated {len(responses)} responses")
        
        # Evaluate responses
        print("Evaluating responses...")
        evaluation_results = evaluate_responses(responses, ground_truths, prompts, args.extraction_method)
        
        # Store trial results
        trial_metrics = evaluation_results["metrics"]
        all_trial_metrics.append(trial_metrics)
        all_trial_results.append({
            "trial": trial_idx + 1,
            "metrics": trial_metrics,
            "detailed_results": evaluation_results["detailed_results"],
            "failed_predictions": evaluation_results.get("failed_predictions", [])
        })
        
        # Print trial metrics
        print(f"\nTrial {trial_idx + 1} Metrics:")
        print(f"  Accuracy:       {trial_metrics['accuracy']:.4f} ({trial_metrics['accuracy']*100:.2f}%)")
        print(f"  Format Rate:    {trial_metrics['format_rate']:.4f} ({trial_metrics['format_rate']*100:.2f}%)")
        print(f"  No Answer Rate: {trial_metrics['no_answer_rate']:.4f} ({trial_metrics['no_answer_rate']*100:.2f}%)")
        
        # Log each trial as a step (not as separate metric names)
        if azureml_run:
            try:
                step = trial_idx  # Use trial index as step
                azureml_run.log("eval/accuracy", trial_metrics['accuracy'], step=step)
                azureml_run.log("eval/format_rate", trial_metrics['format_rate'], step=step)
                azureml_run.log("eval/no_answer_rate", trial_metrics['no_answer_rate'], step=step)
                azureml_run.log("eval/correct_answers", trial_metrics['correct_answers'], step=step)
                print(f"Logged trial {trial_idx + 1} metrics to AzureML as step {step}")
            except Exception as e:
                print(f"Warning: Failed to log trial {trial_idx+1} metrics to AzureML: {e}")
    
    # Calculate aggregate statistics across trials
    print(f"\n[5/5] Computing aggregate statistics across {args.number_of_trials} trial(s)...")
    
    # Extract metric values across all trials
    accuracy_values = [m['accuracy'] for m in all_trial_metrics]
    format_rate_values = [m['format_rate'] for m in all_trial_metrics]
    no_answer_rate_values = [m['no_answer_rate'] for m in all_trial_metrics]
    correct_answers_values = [m['correct_answers'] for m in all_trial_metrics]
    
    # Compute statistics
    aggregate_metrics = {
        "number_of_trials": args.number_of_trials,
        "accuracy": {
            "mean": float(np.mean(accuracy_values)),
            "std": float(np.std(accuracy_values, ddof=1)) if args.number_of_trials > 1 else 0.0,
            "variance": float(np.var(accuracy_values, ddof=1)) if args.number_of_trials > 1 else 0.0,
            "values": accuracy_values
        },
        "format_rate": {
            "mean": float(np.mean(format_rate_values)),
            "std": float(np.std(format_rate_values, ddof=1)) if args.number_of_trials > 1 else 0.0,
            "variance": float(np.var(format_rate_values, ddof=1)) if args.number_of_trials > 1 else 0.0,
            "values": format_rate_values
        },
        "no_answer_rate": {
            "mean": float(np.mean(no_answer_rate_values)),
            "std": float(np.std(no_answer_rate_values, ddof=1)) if args.number_of_trials > 1 else 0.0,
            "variance": float(np.var(no_answer_rate_values, ddof=1)) if args.number_of_trials > 1 else 0.0,
            "values": no_answer_rate_values
        },
        "correct_answers": {
            "mean": float(np.mean(correct_answers_values)),
            "std": float(np.std(correct_answers_values, ddof=1)) if args.number_of_trials > 1 else 0.0,
            "variance": float(np.var(correct_answers_values, ddof=1)) if args.number_of_trials > 1 else 0.0,
            "values": correct_answers_values
        },
        "total_samples": all_trial_metrics[0]['total_samples']
    }
    
    # Print aggregate metrics
    print("\n" + "=" * 80)
    print("AGGREGATE EVALUATION METRICS")
    print("=" * 80)
    print(f"Number of Trials:     {aggregate_metrics['number_of_trials']}")
    print(f"Total Samples:        {aggregate_metrics['total_samples']}")
    print(f"\nAccuracy:")
    print(f"  Mean:               {aggregate_metrics['accuracy']['mean']:.4f} ({aggregate_metrics['accuracy']['mean']*100:.2f}%)")
    if args.number_of_trials > 1:
        print(f"  Std Dev:            {aggregate_metrics['accuracy']['std']:.4f}")
        print(f"  Variance:           {aggregate_metrics['accuracy']['variance']:.6f}")
    print(f"\nFormat Rate:")
    print(f"  Mean:               {aggregate_metrics['format_rate']['mean']:.4f} ({aggregate_metrics['format_rate']['mean']*100:.2f}%)")
    if args.number_of_trials > 1:
        print(f"  Std Dev:            {aggregate_metrics['format_rate']['std']:.4f}")
        print(f"  Variance:           {aggregate_metrics['format_rate']['variance']:.6f}")
    print(f"\nNo Answer Rate:")
    print(f"  Mean:               {aggregate_metrics['no_answer_rate']['mean']:.4f} ({aggregate_metrics['no_answer_rate']['mean']*100:.2f}%)")
    if args.number_of_trials > 1:
        print(f"  Std Dev:            {aggregate_metrics['no_answer_rate']['std']:.4f}")
        print(f"  Variance:           {aggregate_metrics['no_answer_rate']['variance']:.6f}")
    print(f"\nCorrect Answers:")
    print(f"  Mean:               {aggregate_metrics['correct_answers']['mean']:.2f}")
    if args.number_of_trials > 1:
        print(f"  Std Dev:            {aggregate_metrics['correct_answers']['std']:.2f}")
        print(f"  Variance:           {aggregate_metrics['correct_answers']['variance']:.2f}")
    print("=" * 80)
    
    # Log aggregate metrics to AzureML (without step, as final summary)
    if azureml_run:
        print("\nLogging aggregate metrics to AzureML...")
        try:
            azureml_run.log("eval/accuracy_mean", aggregate_metrics['accuracy']['mean'])
            azureml_run.log("eval/format_rate_mean", aggregate_metrics['format_rate']['mean'])
            azureml_run.log("eval/no_answer_rate_mean", aggregate_metrics['no_answer_rate']['mean'])
            azureml_run.log("eval/correct_answers_mean", aggregate_metrics['correct_answers']['mean'])
            azureml_run.log("eval/number_of_trials", aggregate_metrics['number_of_trials'])
            
            if args.number_of_trials > 1:
                azureml_run.log("eval/accuracy_std", aggregate_metrics['accuracy']['std'])
                azureml_run.log("eval/accuracy_variance", aggregate_metrics['accuracy']['variance'])
                azureml_run.log("eval/format_rate_std", aggregate_metrics['format_rate']['std'])
                azureml_run.log("eval/format_rate_variance", aggregate_metrics['format_rate']['variance'])
                azureml_run.log("eval/no_answer_rate_std", aggregate_metrics['no_answer_rate']['std'])
                azureml_run.log("eval/no_answer_rate_variance", aggregate_metrics['no_answer_rate']['variance'])
                azureml_run.log("eval/correct_answers_std", aggregate_metrics['correct_answers']['std'])
                azureml_run.log("eval/correct_answers_variance", aggregate_metrics['correct_answers']['variance'])
            
            print("Successfully logged aggregate metrics to AzureML")
        except Exception as e:
            print(f"Warning: Failed to log aggregate metrics to AzureML: {e}")
    
    # Save results
    print("\nSaving results...")
    
    # Save aggregate metrics
    with open(os.path.join(args.output_dir, "aggregate_metrics.json"), 'w') as f:
        json.dump(aggregate_metrics, f, indent=2)
    
    # Save all trial metrics
    with open(os.path.join(args.output_dir, "all_trials_metrics.json"), 'w') as f:
        json.dump(all_trial_metrics, f, indent=2)
    
    # Save detailed results for each trial
    for trial_idx, trial_result in enumerate(all_trial_results):
        trial_num = trial_idx + 1
        
        with open(os.path.join(args.output_dir, f"trial_{trial_num}_metrics.json"), 'w') as f:
            json.dump(trial_result["metrics"], f, indent=2)
        
        with open(os.path.join(args.output_dir, f"trial_{trial_num}_detailed_results.jsonl"), 'w') as f:
            for result in trial_result["detailed_results"]:
                f.write(json.dumps(result) + '\n')
        
        failed_predictions = trial_result.get("failed_predictions", [])
        if failed_predictions:
            with open(os.path.join(args.output_dir, f"trial_{trial_num}_failed_predictions.jsonl"), 'w') as f:
                for failed in failed_predictions:
                    f.write(json.dumps(failed) + '\n')
    
    # Save summary
    summary = {
        "config": vars(args),
        "aggregate_metrics": aggregate_metrics,
        "all_trials_metrics": all_trial_metrics
    }
    with open(os.path.join(args.output_dir, "summary.json"), 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\nResults saved to: {args.output_dir}")
    print("  - aggregate_metrics.json: Aggregate statistics across all trials")
    print("  - all_trials_metrics.json: Individual metrics for each trial")
    print(f"  - trial_N_*.json/jsonl: Individual trial files for each of {args.number_of_trials} trial(s)")
    print("  - summary.json: Complete summary with config and all metrics")
    print("\nEvaluation completed successfully!")


if __name__ == "__main__":
    main()
